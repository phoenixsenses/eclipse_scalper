# Eclipse Scalper — System State (güncel)

> Bu dosya her önemli değişiklikten sonra güncellenir.
> Codex'e veya yeni session'a direkt paste et — eksiksiz context burada.

**Son güncelleme:** 2026-07-02

---

## 1. Çalışan Prosesler (`start_eclipse.ps1`)

| Role | Script | Notlar |
|---|---|---|
| collector_supervisor | scripts/collector_supervisor.py | microstructure_collector + event_diary'yi o spawn eder |
| heartbeat_watchdog | tools/heartbeat_watchdog.py | 30s interval, 420s max-age |
| bookticker_collector | data/bookticker_collector.py | BTCUSDT/ETHUSDT/SOLUSDT |
| s34_shadow_paper_runner | tools/s34_shadow_paper_runner.py | PAPER_ONLY_NO_ORDERS |
| s34_live_chart | tools/s34_live_chart.py | :5050, read-only |
| s34_v_engine_v02_shadow_mirror | tools/s34_v_engine_v02_shadow_mirror.py | SHADOW_ONLY_NO_ORDERS |
| s34_state_machine_shadow_runner | tools/s34_realtime_shadow_runner.py | STATE_MACHINE_SHADOW_NO_ORDERS |
| **s34_state_machine_live_executor** | tools/s34_state_machine_live_executor.py | **TEK SİLAHLI LIVE EXECUTOR** |
| orderflow_chart | tools/orderflow_chart.py | :5051 |
| s34_replay | tools/s34_replay.py | :5052 |

Restart: `powershell -File start_eclipse.ps1`
Stop: `powershell -File stop_eclipse.ps1`
Status: `powershell -File status_eclipse.ps1`

---

## 2. Live Executor Sabitleri (tools/s34_state_machine_live_executor.py)

```python
RULE_NAME              = "S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3"
SYMBOL                 = "ETHUSDT"
LIQ_SIDE               = "SELL"
THRESHOLD_USD          = 200_000.0      # ETH SELL anchor eşiği
PROP_THRESH_USD        = 50_000.0       # follow-on / propagation
BTC_CONFIRM_USD        = 2_000_000.0   # BTC SHORT confirm eşiği ← güncel
BTC_CONFIRM_MIN_DELAY_MS = 5 * 60_000  # T+5dk sonrası BTC aranır ← güncel
BTC_7D_LOOKBACK_MS     = 7 * 24 * 3600_000
SIL_LO_MS             = 60_000
SIL_HI_MS             = 30 * 60_000
LONG_HORIZON_MS        = 4 * 3600_000
SHORT_HORIZON_MS       = 2 * 3600_000
DEFAULT_STOP_BPS       = 150.0
```

### LONG gate (long_eligible):
```python
long_eligible = (
    (not bull)
    and sess != "EUROPE"
    and not (sess == "US" and hour in {13, 14})   # US 13-14 UTC blok ← güncel
    and dow not in {0, 2}                           # Mon/Wed blok
    and sync_k < 200_000.0                          # sync gate ← güncel
    and (float(btc4h) < 0.0 or float(btc7d) < 0.0)  # BTC 4h OR 7d negatif ← güncel
    and (base_score + 1) >= 3
)
```

### SHORT gate (short_eligible):
```python
short_eligible = (not bull) and sess != "EUROPE" and dow != 6 and base_score >= 4
# SHORT açılır sadece: BTC SELL ≥2M, T+5dk delay sonrası
```

### Score bileşenleri (base_score, max 5):
```
n2h >= 3        (+1)
btc4h < 0       (+1)
vdepth >= 30    (+1)
sess == "US"    (+1)
sync_k >= 200K  (+1)
```
`long_score = base_score + 1` (silence varsayımı)

---

## 3. Shadow Runner Sabitleri (tools/s34_realtime_shadow_runner.py)

```python
ETH_THRESH              = 200_000.0
PROP_THRESH             = 50_000.0
BTC_THRESH              = 2_000_000.0    # ← güncel
BTC_CONFIRM_MIN_DELAY_MS = 5 * 60_000   # ← güncel
BTC_7D_LOOKBACK_MS      = 7 * 24 * 3600_000
SIL_LO_MS              = 60_000
SIL_HI_MS              = 30 * 60_000
HORIZON_LONG_MS         = 4 * 3600_000
HORIZON_SHORT_MS        = 2 * 3600_000
FEE_BPS                 = 5.0
```

Gate'ler live executor ile aynı (her ikisi de senkron).

---

## 4. Araştırma Sonuçları — Key Numbers

### LONG (TIME_EXIT population, N=72, base)
| Filtre | N | WR | Avg bps | /ay |
|---|---|---|---|---|
| Tüm TIME_EXIT | 72 | 68.1% | +47.9 | — |
| + sync<200K | 34 | 70.6% | +76.2 | 8.5 |
| + sync<200K + excl US 13-14 | ~28 | ~80% | +98 | ~7 |
| + sync<200K + btc7d<0 | ~20 | ~85% | +111 | ~5 |
| + sync<200K + excl US 13-14 + btc7d<0 | 12 | **91.7%** | **+151.7** | 3.0 |

### SHORT (base_score ≥ 4)
| Filtre | N | WR | Avg bps | /ay |
|---|---|---|---|---|
| score ≥ 4 | 15 | 80% | +116.6 | 7.5 |
| score ≥ 4 + BTC ≥ 2M | ~5 | 100% | +198.5 | 2.5 |
| score ≥ 4 + BTC ≥ 2M + delay ≥ 5m | 3 | 100% | +250.5 | ~1.5 |

### Bilinen riskler
- Tail: trade'lerin %18.7'si −100+ bps kaybeder
- Worst real fill: −175.7 bps (stop gap-through)
- Atomicity: entry → stop arası ~2s unprotected window
- Oversize: şu an ~60x tail-budget üzerinde (**URGENT**)

---

## 5. Execution Management Panel (dashboard)

Dosya: `tools/s34_cascade_navigation_dashboard.py`
Output: `reports/research/s34/S34_CASCADE_NAVIGATION_DASHBOARD.json` + `.md`

Paneller:
- **EXECMGMT**: notional vs tail-budget, oversize multiple, recommended max margin
- **STOPPROT**: nominal stop vs worst fill, gap-through bps, atomicity warning
- **MGMT**: live state, kill status, regime
- **EXECAUD**: audit panel (99.3x oversize — farklı formül, STOPPROT referansı)

Equity kaynağı (öncelik sırası):
1. `.env` ACCOUNT_EQUITY_USD vb. (yok)
2. `notional / leverage / (margin_pct / 100)` → `equity_source: derived_notional/lev/margin_pct(85%)`
3. Hardcoded $35 fallback

---

## 6. Kritik Dosyalar

| Dosya | Açıklama |
|---|---|
| `tools/s34_state_machine_live_executor.py` | **Live executor — DOKUNMA** |
| `tools/s34_realtime_shadow_runner.py` | Shadow runner (live ile senkron) |
| `tools/s34_live_chart.py` | Dashboard frontend (:5050) |
| `tools/s34_cascade_navigation_dashboard.py` | Risk/mgmt panel (read-only) |
| `tools/s34_shadow_paper_runner.py` | Eski V-engine paper runner |
| `data/microstructure.db` | SQLite — tüm likidasyon + mark price verisi |
| `runtime/s34_v_engine_live_state.json` | Live executor state (active, orders, pending) |
| `reports/shadow/s34_state_machine_shadow.jsonl` | Shadow ledger |
| `.env` | API keys + live config — **DOKUNMA** |

---

## 7. Kesin Guardrail'lar (Codex dahil hiç kimse ihlal edemez)

1. `tools/s34_state_machine_live_executor.py` — order logic, sizing, leverage, .env → operator sign-off olmadan değiştirme
2. `.env` dosyası — okuma tamam, yazma yasak
3. `execution/`, `risk/`, `brain/` klasörleri — dokunma
4. Leverage = **15x** — operatör kararı 2026-07-01 (40x→15x düşürüldü; likidasyon ~%6.5, stop'lar likidasyondan önce tetiklenir). Yeniden değiştirme = operatör sign-off.
5. ORDER_NOTIONAL_USD / position size — değişmez
6. Paralel Python/PS prosesi çalıştırma — RAM yetmez, çökme
7. pytest'te 2'den fazla test dosyası — RAM çökme
8. Lookahead kullanma — DAT-01 compliance
9. D:\ drive kullanımı — kurulumlar ve projeler D:'de

---

## 8. Mevcut Durum / Sonraki Adımlar

**Tamamlanan (bu session):**
- [x] sync<200K LONG gate — live + shadow
- [x] score≥4 SHORT gate — live + shadow
- [x] US 13-14 UTC blok — live + shadow
- [x] btc7d<0 gate — live + shadow
- [x] BTC confirm 1M → 2M — live + shadow
- [x] BTC delay 0 → 5dk — live + shadow
- [x] EXECMGMT + STOPPROT dashboard paneli
- [x] Equity derivation (dynamic, not hardcoded)
- [x] Duplicate live executor fix — `collector_supervisor.py` artık live executor spawn etmez; `start_eclipse.ps1` duplicate process temizler/idempotent çalışır

**Operatör aksiyonu gereken (Codex yapamaz):**
- [ ] Per-trade margin'i tail-budget'a indir (~$0.50 max, şu an $29.8) — URGENT

**Araştırma backlog:**
- ETH4h<-100 gate (WR 87.5%, N=16 — 4.0/ay) — önce daha fazla data bekle
- sync=0 filtreleme (2 event, her ikisi negatif) — zaten az event, düşük öncelik
- Forward OOS validation (30-60 gün shadow ledger review)

---

## 9. Son Frekans Artırma Testleri (2026-07-01)

Rapor dosyaları:
- `reports/research/s34/S34_FREQ_EXPANSION_TESTS.md`
- `reports/research/s34/S34_QUESTION_TESTS_A1_A10.md`
- JSON eşleri aynı klasörde.

Not: Bu run, daha geniş `200K NAV-matched TIME_EXIT + sync<200K` universe ile koşuldu.
Bu yüzden eski dar live-like N=34 baseline ile birebir aynı değil.

### Güncel offline baseline
| Universe | N | WR | Avg bps | /ay |
|---|---:|---:|---:|---:|
| TIME_EXIT + sync<200K | 100 | 58.0% | +26.5 | 23.0 |

### En önemli frequency bulguları
| Test | N | WR | Avg bps | /ay | Karar |
|---|---:|---:|---:|---:|---|
| B1 btc3d<0 | 9 | 100.0% | +163.7 | 2.3 | PROMISING ama düşük N |
| B2 btc4h<0 only, btc7d yok | 17 | 82.4% | +94.7 | 4.3 | PROMISING |
| B3 ASIA + sync<200K + btc7d<0 | 12 | 75.0% | +93.8 | 3.0 | PROMISING |
| B8 SHORT BTC>=1M + delay>=10m | 10 | 90.0% | +133.2 | 2.7 | PROMISING |
| B8 current SHORT BTC>=2M + delay>=5m | 7 | 85.7% | +99.0 | 1.9 | Current conservative |
| B10 btc7d<0 OR score5 | 9 | 100.0% | +170.6 | 2.3 | Same low-frequency pocket |
| B5 ETH SELL 150K added band | 143 | 55.2% | +7.0 | 32.8 | WORSE |

### A-test karar notları
- A1 DOW: Wednesday block destekli (N=17, WR 35.3%, avg -5.9). Monday block geniş universe'de güçlü destekli değil (N=18, WR 66.7%, avg +32.4).
- A2 sync curve: sync threshold tek başına temiz edge değil. sync<300K veya <500K daha fazla N veriyor ama WR 60% civarı.
- A3 btc7d curve: btc7d gevşetmek ana frequency lever. `btc7d<+500` N=14, WR 85.7%, avg +123.4; all N=21, WR 76.2%, avg +82.0.
- A4 US 13/14 block: korunmalı. Blocked US13/14 subset N=3, WR 33.3%, avg -3.8.
- A5 score relaxation: base_score1 ek eventları N=6, WR 83.3%, avg +98.3. Frequency için umutlu ama gauntlet gerekir.
- A6 n2h>=2 relaxation: sadece 3 event ekliyor, hepsi winner ama avg düşük (+27.8).
- A7 vdepth>=20/25 relaxation: sadece 1 zayıf event ekliyor; düşük öncelik.
- A8 noisy early exit: early exit realized avg -28.4; 4h counterfactual avg -24.2. Noisy eventler alpha değil; early exit tek başına kurtarmıyor.
- A10 running notional sweet spot: 300K-500K bandı en iyi (N=28, WR 71.4%, avg +53.0). Extreme 1M+ iyi değil.

### En mantıklı sonraki gauntlet adayları
1. LONG frequency relax: mevcut gate'ten `btc7d<0` şartını gevşetme veya kaldırma, ama `btc4h<0` / `btc3d<0` ile karşılaştır.
2. LONG score relaxation: base_score1 eklemek. In-sample umutlu; holdout/permutation olmadan live değil.
3. SHORT frequency: `BTC>=1M + delay>=10m` adayını current `BTC>=2M + delay>=5m` ile karşılaştır.
4. Combined target: LONG relaxed gate + SHORT BTC1M delay10 kabaca 8/month potansiyel verir; live'a geçmeden önce chronological holdout + MC/permutation + real-cost gauntlet gerekir.

### Net guardrail
Bu testler research-only. `tools/s34_state_machine_live_executor.py`, `.env`, order logic, leverage ve sizing değiştirilmedi.

---

## 11. Process / Startup Durumu (2026-07-01)

Duplicate live executor problemi çözüldü.

Değişiklikler:
- `scripts/collector_supervisor.py`: `tools.s34_state_machine_live_executor` managed process listesinden çıkarıldı. Supervisor artık live executor spawn etmez; sadece collector + event diary yönetir.
- `start_eclipse.ps1`: `Start-RegisteredPythonProcess` aynı command needle için duplicate Python process bulursa tek PID bırakır ve fazlaları `STOPPED_DUPLICATE` olarak kapatır.

Doğrulama:
- `powershell -NoProfile -ExecutionPolicy Bypass -File start_eclipse.ps1` temiz restart edildi.
- `powershell -NoProfile -ExecutionPolicy Bypass -File start_eclipse.ps1 -NoCleanStop -StatusWaitSec 1` idempotency test geçti: tüm roller `ALREADY_RUNNING`.
- Tek live executor PID: `19112`.
- `logs/pids/s34_state_machine_live_executor.pid` ve `logs/pids/s34_v_engine_live_executor.pid` ikisi de `19112`.
- Status: collector ok, bookticker ok, watchdog GREEN, live state age ~17s, active null, orders 0.

Not:
- Normal sandbox içinden start etmek collector/bookticker network erişimini bozabilir. Gerçek restart için `start_eclipse.ps1` sandbox dışı / normal PowerShell yetkisiyle çalıştırılmalı.

---

## 10. Next Candidate Gauntlet (2026-07-01)

Rapor dosyaları:
- `reports/research/s34/S34_NEXT_CANDIDATE_GAUNTLET.md`
- `reports/research/s34/S34_NEXT_CANDIDATE_GAUNTLET.json`

Test kapsamı:
- LONG gate relaxation: btc7d gevşetme/kaldırma, btc4h/btc3d alternatifleri, Mon block kaldırma, base_score1 ekleme, notional 300K-500K.
- SHORT expansion: BTC confirm 1M/2M, delay 5/10/15m, hold 90/120/150/180m.
- State sequence: silence/noisy ve BTC confirm kombinasyonları.
- Robustness: overall stats, 5-fold chronological walk-forward, top-3-removed, family-level max-stat MC permutation.

### En iyi combined adaylar
| Candidate | N | WR | Avg bps | T3R | /ay | WF +sum | WF +T3R | MC p | Not |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C_score_relax_short1m10 | 25 | 92.0% | +138.3 | +2384.8 | 6.2 | 5/5 | 4/5 | 0.018 | En güçlü research adayı |
| C_btc7d500_short1m10 | 24 | 87.5% | +127.5 | +2015.2 | 6.1 | 5/5 | 3/5 | 0.240 | Güçlü ama MC geçmedi |
| C_no_btc7d_short1m10 | 31 | 80.6% | +98.5 | +2008.8 | 7.9 | 5/5 | 3/5 | 0.248 | 8/month hedefe en yakın |
| C_freq_balanced_btc4h_short1m10 | 27 | 85.2% | +108.9 | +1897.1 | 6.9 | 5/5 | 3/5 | 0.375 | Dengeli ama MC geçmedi |
| C_current_live_long_short | 16 | 93.8% | +139.3 | +1184.8 | 4.1 | 4/5 | 3/5 | 0.977 | Current conservative, düşük frekans |

### En iyi tek bacaklar
- LONG base_score1_added: N=15, WR=93.3%, avg +141.7, T3R +1082.6, 5/5 fold pozitif, MC p=0.109. Frequency için umutlu ama tek başına MC geçmedi.
- LONG btc7d<500: N=14, WR=85.7%, avg +123.4, T3R +782.1, 5/5 fold pozitif, MC p=0.636.
- LONG btc4h<0 no btc7d: N=17, WR=82.4%, avg +94.7, T3R +751.0, 4/5 fold sum pozitif, MC p=0.798.
- SHORT btc1m delay5: N=12, WR=83.3%, avg +123.3, T3R +558.8, 5/5 fold pozitif, MC p=0.236.
- SHORT btc1m delay10: N=10, WR=90.0%, avg +133.2, T3R +523.6, 4/5 fold pozitif, MC p=0.440.

### State sequence sonucu
- `SEQ_silence_no_btc1m`: N=166, WR=60.8%, avg +37.9, T3R +4947.3, MC p=0.001. Bu alpha değil ama navigation etiketi olarak güçlü: BTC confirm olmayan silence state daha sağlıklı.
- `SEQ_noisy_with_btc1m`: N=48, WR=22.9%, avg -129.4, T3R -6615.0. BTC confirm ile noisy/follow-on birlikteyse LONG için açık danger state.
- `SEQ_noisy_no_btc1m`: N=205, WR=57.1%, avg -1.3, T3R -1341.7. Noisy state genel olarak LONG alpha değil.

### Karar
- Live'a hemen alınan değişiklik yok.
- En güçlü yeni research adayı: `C_score_relax_short1m10` = LONG base_score1 relaxation + SHORT BTC>=1M delay>=10m.
- Frequency hedefi için en pratik aday: `C_no_btc7d_short1m10` = btc7d gate kaldırılmış LONG + SHORT BTC>=1M delay>=10m; 7.9/month ama MC geçmedi.
- Sonraki live öncesi şart: daha dar live-like universe üzerinde tekrar, chronological holdout, permutation, forward-shadow. Bu run research-only.

---

## 12. State Machine Deep-Dive Tests (2026-07-01)

Rapor dosyaları:
- `reports/research/s34/S34_STATE_MACHINE_DEEP_DIVE_TESTS.md`
- `reports/research/s34/S34_STATE_MACHINE_DEEP_DIVE_TESTS.json`

Test edilen fikirler:
1. Relaxed adayların current live combo'ya göre added-only katkısı.
2. 70/30 chronological holdout.
3. Month stability / regime dependence.
4. Fee sensitivity (5 bps bazdan 8/10/15 bps maliyete).
5. Tail/drawdown: worst loss, -100 bps tail, max DD, loss streak.
6. No-overlap execution: live gibi tek pozisyon varsayımı.
7. Candidate overlap map.
8. State navigation: silence/noisy + BTC confirm OK/DANGER etiketleri.
9. Current-vs-relaxed delta.
10. Live-readiness score.

### Ana sonuç tablosu
| Candidate | N | WR | Avg bps | T3R | Added N | Added Avg | Holdout Sum | Holdout T3R | Worst | TailN | NoOverlap N | Readiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C_current_live_long_short | 16 | 93.8% | +139.3 | +1184.8 | 0 | — | +358.9 | -154.3 | -169.6 | 1 | 13 | RESEARCH_ONLY_LOW_N |
| C_score_relax_short1m10 | 25 | 92.0% | +138.3 | +2384.8 | 12 | +97.5 | +1223.3 | +331.8 | -49.2 | 0 | 20 | PAPER_CANDIDATE |
| C_no_btc7d_short1m10 | 31 | 80.6% | +98.5 | +2008.8 | 18 | +42.5 | +1472.4 | +580.9 | -197.9 | 1 | 25 | PAPER_SHADOW_ONLY_TAIL |
| C_freq_balanced_btc4h_short1m10 | 27 | 85.2% | +108.9 | +1897.1 | 16 | +62.0 | +1395.7 | +504.2 | -33.8 | 0 | 22 | PAPER_CANDIDATE |
| C_btc7d500_short1m10 | 24 | 87.5% | +127.5 | +2015.2 | 11 | +70.2 | +1309.9 | +418.4 | -25.5 | 0 | 20 | PAPER_CANDIDATE |

### Navigation state sonucu
- `SEQ_silence_no_btc1m`: N=166, WR=60.8%, avg +37.9, T3R +4947.3. Broad OK-state etiketi; trade alpha olarak değil, navigation context olarak kullan.
- `SEQ_noisy_with_btc1m`: N=48, WR=22.9%, avg -129.4, T3R -6615.0, TailN=23. LONG için güçlü DANGER-state.

### Karar
- En temiz yeni aday: `C_score_relax_short1m10`. Added-only +97.5 bps, holdout T3R pozitif, -100 bps tail yok, no-overlap sonrası N=20.
- En yüksek frequency adayı: `C_no_btc7d_short1m10`, ama -197.9 bps tail var; risk nedeniyle sadece shadow/paper.
- `C_freq_balanced_btc4h_short1m10` ve `C_btc7d500_short1m10` iyi ikinci adaylar; ikisi de paper-candidate seviyesinde ama live için forward gerekir.
- Current live combo küçük N ve holdout T3R negatif olduğu için büyütme/ekleme yok.
- Live'a değişiklik yapılmadı; sıradaki doğru adım leading adayları ayrı shadow bucket olarak izlemek.

---

## 13. Dashboard Candidate Tracker + Sonraki Test Kuyruğu (2026-07-01)

### Dashboard'daki 3 canlı bucket (tools/s34_live_chart.py)

| Bucket | Filtre | N (backfill) | WR | Avg bps | MC p | Not |
|---|---|---:|---:|---:|---:|---|
| C_score_relax | LONG_SILENCE + TIME_EXIT + sync<200K + score≥2 **(btc7d gate kaldırıldı)** | 34 | 70.6% | +76.2 | 0.018 | btc7d backfill'de yok; score bazında tracked |
| C_btc7d500 | LONG_SILENCE + TIME_EXIT + sync<200K + btc7d∈(-500,0) | 0 | — | — | 0.240 | live event'ler biriktikçe dolacak |
| C_freq_btc4h | LONG_SILENCE + TIME_EXIT + sync<200K + btc4h<0 (no btc7d) | 0 | — | — | 0.375 | live event'ler biriktikçe dolacak |

> **Not:** Shadow runner'a `btc4h_bps` eklendi (2026-07-01). Yeni CLOSE event'lerinden itibaren her iki field de (`btc7d_bps`, `btc4h_bps`) loglanır. Backfill event'lerinde yok.

### SEQ_noisy_with_btc1m → Tersine SHORT fırsatı

LONG pozisyon için T3R=-6615.0 olan danger state; tersi potansiyel SHORT sinyali:
- Koşul: ETH SELL ≥200K → noisy follow-on ≥50K (1-30dk) + BTC ≥1M (herhangi zamanda, delay≥5dk)
- Observed: N=48, LONG WR=22.9%, avg=-129.4 bps
- Tersine SHORT: ~WR≈77%, avg≈+129.4 bps (tahmin — backtest lazım)
- Araştırma adı: `SHORT_NOISY_BTC1M`
- Mevcut durumda bu path `if False` ile kapalı (SHORT_NOISY satırı shadow runner'da)
- **Henüz backtest yapılmadı — research-only hipotez**

### Sonraki test kuyruğu (öncelik sırasıyla)

| Öncelik | Test | Açıklama | Araç |
|---|---|---|---|
| 1 | SEQ_noisy_with_btc1m SHORT | N=48 LONG danger → SHORT extraction. Noisy ETH + BTC≥1M →SHORT. Backtest: WR, avg, T3R, holdout, MC. | Yeni script |
| 2 | C_score_relax gauntlet | base_score1 LONG + SHORT BTC≥1M delay10m. Dar live-like universe üzerinde chronological holdout + MC. | Codex |
| 3 | C_btc7d500 gauntlet | btc7d∈(-500,0) gate LONG + SHORT combo. Same gauntlet. | Codex |
| 4 | btc4h vs btc7d karşılaştırması | btc4h<0 tek başına vs btc7d<0 — hangisi daha stabil? Regime-split test. | Codex |
| 5 | Forward OOS validation | 3 dashboard bucket'ı shadow ledger üzerinden 30-60 gün izle. Yeterli N gelince gauntlet tekrar. | Manual/zaman |
| 6 | SHORT_NEITHER history | Gerçek CLOSE event'ler için SHORT_NEITHER route'unu ölçüm başlat. N=0 şu an. | Shadow runner zaten aktif |

### Operator aksiyonu (Codex yapamaz — URGENT)
- Per-trade margin: $29.8 → ~$0.50 (tail-budget). 60x oversize hâlâ aktif. Stop var ama gap-through riski sürüyor.

---

## 14. Mega Araştırma Sonuçları — A-F Kapsamlı Test (2026-07-01)

Rapor: `reports/research/s34/S34_MEGA_V1.json` + `.md`
Evren: 596 ETH SELL anchor, 4.5 ay, tek process.

### En Güçlü Yeni Bulgular

| Bulgu | Kod | N | WR | Avg bps | MC p | Karar |
|---|---|---:|---:|---:|---:|---|
| Double cascade setup (density+prebuildup) | D8 | 12 | 100% | +178.2 | 0.001 | **BUCKET AÇ** |
| BTC also falling fast (btc5m<-20) | D5 | 15 | 80% | +109.8 | 0.001 | **BUCKET AÇ** |
| Failed cascade → LONG | D1 | 23 | 78.3% | +82.8 | 0.006 | **BUCKET AÇ** |
| Echo cascade (2nd in 45min) | B5 | 12 | 83.3% | +108.0 | 0.026 | Research only |
| High prebuildup (>=3 events) | C5 | 5 | 100% | +233.8 | 0.034 | Research only (az N) |
| btc3d<0 gate (btc7d yerine) | A3 | 43 | 74.4% | +62.2 | 0.0 | Gate swap testi yap |
| -50 bps stop: avg +51.7→+74.6 | E4 | 39 | 69.2% | +74.6 | 0.0 | Operatör kararı |
| Ask-heavy order book | F1 | 11 | 81.8% | +64.1 | 0.029 | Research only |

### Kapatılan Hipotezler (geçmedi)

| Hipotez | Bulgu | Karar |
|---|---|---|
| ETH BUY cascade → SHORT | WR=46%, avg=-18.5, MC p=0.988 | ❌ Kesin KAPALI |
| OFF session (ASIA) LONG | MC p=0.264, WR=57.9% | ❌ Anlamsız |
| SOL lead gate | N=0 (SOL notional yok) | ❌ Veri yok |
| Gerçek cascade (fiyat T+5m aşağı) | WR=56.2%, avg=+6.9, MC p=0.416 | ❌ Anlamsız |
| Partial exit (2h+4h) | avg=+38.6 vs baseline +51.7 | ❌ Full 4h daha iyi |
| Vol regime filtresi | tüm MC p>0.07 | ❌ Anlamsız |
| Cumartesi sabahı LONG | WR=14.3%, avg=-90.8 | ❌ Onaylandı: çok kötü |

### Tehlike Sinyalleri (VETO adayları)

| Sinyal | Bulgu | Öneri |
|---|---|---|
| BTC/ETH ratio >= 2x | N=2, WR=0%, avg=-299.5 bps | BTC cascade >> ETH cascade → LONG girme |
| Cumartesi sabahı (dow=5, h<12) | WR=14.3%, avg=-90.8 | Zaten Cumartesi blokta ama daha erken başlat |
| Yüksek prebuildup YOK (prebuildup<3) | WR=64.7%, MC p=0.145 | Düşük prebuildup = güvenilmez |

### Yeni Dashboard Bucket Tanımları

```
DOUBLE_CASCADE: base_long(ev) AND density_24h>=1 AND prebuildup>=2
BTC_FALLING_FAST: base_long(ev) AND btc5m<-20 bps
FAILED_CASCADE: base_long(ev) AND failed_cascade=True (T+5m price UP)
```

> Not: Bu 3 bucket mevcut `tools/s34_live_chart.py` candidate tracker'ına eklendi (2026-07-01).
> `tools/s34_realtime_shadow_runner.py` bu field'ları (btc5m_bps, density_24h, prebuildup, failed_cascade) live eventlerden itibaren loguluyor.

---

## 15. Frekans Genişletme Araştırması — 20-30 Trade/Ay Hedefi (2026-07-01)

Rapor: `reports/research/s34/S34_FREQ_EXPANSION_V1.json` + `.md`
Evren: 596 ETH SELL anchor, 4.5 ay. Script: `tools/research_s34_freq_expansion_v1.py`

### Ana Bulgular

**A — ETH Eşik Taraması** (minimal gate: not bull, not EU, score>=2)

| Eşik | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| 50K | 600 | 132.8 | 58.7% | +16.4 | 0.005 |
| 75K | 511 | 113.1 | 57.9% | +12.5 | 0.033 |
| 150K | 376 | 83.2 | 60.1% | +14.9 | 0.036 |
| 200K (current) | 329 | 72.8 | 59.0% | +12.0 | 0.080 |
| 300K | 254 | 56.2 | 60.6% | +18.6 | 0.035 |

> Sonuç: Düşük eşik (50-150K) edge VERİYOR ama zayıf (+12-17 bps). 200K zaten optimal nokta değil — 300K daha temiz. Eşik düşürme çözüm değil.

**B — Gate Gevşetme (ETH 200K evreninde)**

| Config | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| Mevcut (btc7d<0) | 31 | 6.9 | 74.2% | +81.7 | 0.001 |
| btc4h<0 only | 63 | 13.9 | 73.0% | +50.8 | 0.0 |
| **btc4h<0 OR btc7d<0** | **68** | **15.0** | **73.5%** | **+55.0** | **0.0** |
| No btc7d gate | 74 | 16.4 | 68.9% | +45.0 | 0.0 |
| No BTC + No DOW | 109 | 24.1 | 64.2% | +39.2 | 0.001 |
| btc4h<0 + Sat block | 58 | 12.8 | 75.9% | +60.0 | 0.0 |

> **btc4h<0 OR btc7d<0**: N'i 31→68 (x2.2) yaparken WR 74%→73.5% (minimal kayıp). En iyi gate relax.

**C — SHORT_NOISY Portföy**

| Config | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| BTC>=500K delay5 | 42 | 9.3 | 66.7% | +81.0 | 0.0 |
| **BTC>=1M delay5** | **25** | **5.5** | **76.0%** | **+129.2** | **0.0** |
| BTC>=2M delay5 | 12 | 2.7 | 91.7% | +159.7 | 0.002 |
| BTC>=500K delay5 sc>=3 | 29 | 6.4 | 75.9% | +112.4 | 0.0 |
| BTC>=1M delay5 sc>=3 | 20 | 4.4 | 80.0% | +148.7 | 0.0 |

> BTC>=500K: 9.3/ay, WR=66.7% — daha düşük eşikle frekans artar ama kalite düşer.
> BTC>=1M: 5.5/ay, WR=76.0% — mevcut optimum eşik.

**D — BTC-led ETH Sinyali (YENİ KAYNAK — KAPANDI)**

| Signal | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| BTC>=1M → ETH LONG 4h | 102 | 22.6 | 50.0% | -12.9 | 0.772 |
| BTC>=1M → ETH SHORT 2h | 102 | 22.6 | 44.1% | +11.6 | 0.185 |
| BTC>=500K → ETH LONG 4h (no ETH casc) | 80 | 17.7 | 60.0% | +27.2 | 0.050 |

> ❌ **BTC-led ETH sinyali KAPANDI.** BTC SELL cascade sonrası ETH hiçbir yönde güvenilir edge vermiyor. 20-30/ay hedefi buradan gelemiyor.

**E — SOL-led ETH Sinyali (KAPANDI)**

| Signal | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| SOL>=100K → ETH LONG 4h | 98 | 21.7 | 56.1% | -4.5 | 0.575 |
| SOL>=200K → ETH LONG 4h | 56 | 12.4 | 58.9% | +15.1 | 0.246 |

> ❌ **SOL-led KAPANDI.** SOL cascade ETH için anlamlı sinyal değil.

**F — Echo Cascade Standalone**

| Config | N | /ay | WR | Avg bps | MC p |
|---|---:|---:|---:|---:|---:|
| echo_20_60 (all) | 165 | 36.5 | 59.4% | +26.1 | 0.012 |
| **echo_20_60 + silence** | **54** | **11.9** | **70.4%** | **+75.2** | **0.0** |
| echo_20_60 + gated | 29 | 6.4 | 69.0% | +84.9 | 0.0 |
| echo_30_90 + silence | 76 | 16.8 | 68.4% | +63.1 | 0.0 |

> echo_30_90 + silence: 16.8/ay, WR=68.4% — geniş pencere daha fazla N veriyor.

**G — Kombine Portföy Simülasyonu (non-overlapping)**

| Portföy | N | /ay | WR | Avg bps | T3R | MC p |
|---|---:|---:|---:|---:|---:|---:|
| LONG only (btc4h OR 7d) | 68 | 15.0 | 73.5% | +55.0 | +2772 | 0.0 |
| **LONG + SHORT_NOISY** | **92** | **20.4** | **73.9%** | **+74.4** | **+5771** | **0.0** |
| LONG + Echo silence | 86 | 19.0 | 69.8% | +55.8 | +3747 | 0.0 |
| **LONG + SHORT + Echo** | **110** | **24.3** | **70.9%** | **+71.8** | **+6815** | **0.0** |
| LONG(btc4h_only) + SHORT | 87 | 19.3 | 73.6% | +72.4 | +5228 | 0.0 |
| LONG(no_btc7d) + SHORT | 98 | 21.7 | 70.4% | +65.6 | +5362 | 0.0 |

### 20-30/ay Yol Haritası

| Adım | Sinyal yolu | +/ay | Birikimli | WR |
|---|---|---:|---:|---:|
| 0 | Mevcut live LONG_SILENCE | — | ~7 | 74% |
| 1 | + LONG gate: btc4h<0 OR btc7d<0 | +8 | ~15 | 73.5% |
| 2 | + SHORT_NOISY BTC>=1M delay5 | +5 | ~20 | 74% (combined) |
| 3 | + Echo cascade 30-90m silence | +5 | ~25 | 71% (combined) |
| **Toplam hedef** | | | **~25/ay** | **~71-74%** |

### Kapatılan Hipotezler

- ❌ **BTC-led ETH**: BTC>=1M sonrası ETH hiç yön tutmuyor (WR=50%, MC p=0.77)
- ❌ **SOL-led ETH**: SOL cascade sonrası ETH edge yok (MC p=0.57)
- ❌ **ETH eşik düşürme**: 100-150K edge var ama +10-15 bps — fee'den sonra marginal

### Sıradaki Adımlar (öncelik)

1. **SHORT_NOISY aktive** — shadow runner'daki `if False` kaldır, BTC>=500K delay5 threshold ile (9.3/ay)
2. **LONG gate: btc4h<0 OR btc7d<0** — live executor gate değişikliği (operator sign-off gerekli)
3. **Echo 30-90m silence** — yeni sinyal yolu olarak shadow runner'a ekle
4. Holdout OOS ve shadow accumulation sonrası karar: live promotion veya devam observe

---

## 17. S34 Ideas V2 Test Sonuçları (2026-07-01)

Rapor: `reports/research/s34/S34_IDEAS_V2.json` + `.md`
Script: `tools/research_s34_ideas_v2.py`
Evren: 596 ETH SELL anchor, 4.5 ay, current live gate (btc7d<0) baseline N=31

### Güçlü Yeni Bulgular (mc_p <= 0.05)

| Bulgu | N | WR | Avg bps | mc_p | Öneri |
|---|---:|---:|---:|---:|---|
| **Echo 45-120m + silence** | 17 | 94.1% | +115.2 | 0.0 | BUCKET AÇ — en güçlü echo pencere |
| **Echo 30-90m + OR gate + silence** | 15 | 93.3% | +120.4 | 0.0 | Echo route'u OR gate ile güçlü |
| **State: silence + no BTC (gated)** | 12 | 91.7% | +152.1 | 0.001 | Mevcut stratejinin en temiz state'i |
| **Echo 60-180m + OR gate + silence** | 19 | 89.5% | +113.2 | 0.0 | Daha geniş pencere, daha fazla N |
| **Echo 20-60m + OR gate + silence** | 11 | 90.9% | +127.5 | 0.001 | Dar ama yüksek kalite |
| **OFI silence + post-buyers** | 9 | 100% | +165.4 | 0.001 | Silence + OFI>0 sonra = mükemmel (az N) |
| **LAG: BTC5m below median** | 15 | 86.7% | +134.8 | 0.0 | BTC da düşüyorsa LONG daha güçlü |
| **Vol decile 1-3 (low) at cascade** | 18 | 83.3% | +109.9 | 0.0 | Düşük vol_decile = compression = güçlü |
| **EX: prebuildup > 0 (any)** | 19 | 78.9% | +112.5 | 0.0 | Herhangi prebuildup varsa daha iyi |
| **VAC: quiet 0-15m (low notional)** | 15 | 80.0% | +119.1 | 0.001 | Düşük post-cascade likidasyon = güçlü |
| **SH: SHORT BTC1M 4h hold** | 32 | 71.9% | +109.4 | 0.001 | 4h hold 2h'tan daha iyi avg |
| **SH: SHORT BTC1M 3h hold** | 32 | 75.0% | +95.9 | 0.001 | En iyi WR (3h) |
| **SHORT score>=3 (relaxed gate)** | 24 | 70.8% | +119.1 | 0.0 | score4 yerine score3 = +11 trade /ay |
| **FB: failed_cascade LONG** | 20 | 80.0% | +112.9 | 0.001 | LONG > SHORT için failed cascade |

### Kapanan Hipotezler (bu session)

| Hipotez | Sonuç | Karar |
|---|---|---|
| Failed cascade → SHORT | WR=33%, avg=-27bps, N=335 | ❌ KESİN KAPALI |
| Failed bounce + noisy → SHORT | WR=39%, avg=-18bps | ❌ KAPALI |
| Failed bounce + BTC1M → SHORT | WR=45.5%, mc_p=0.156 | ❌ KAPALI |
| Cascade density filtresi (cur gate) | DEN_0: N=0, DEN_1: N=1 | ❌ ANLAMSIZ (cur gate zaten dense period'da) |
| BTC led (delta<0 vs ETH) | N=4, mc_p=0.132 | ❌ Çok az N, kapandı |
| No BTC nearby = vacuum (LAG_no_btc) | WR=82.6%, N=23 — ama "no BTC" zaten silence in disguise | Mevcut gate zaten bunu yakalar |
| State transition timing (noisy 1-30m) | Tüm timing buckets avg negatif | ❌ Noisy timing önemli değil — noisy genel olarak LONG için kötü |

### Önemli Şaşırtıcı Bulgular

1. **Vol_decile contradiction:** Daha önceki Mega testinde "vol regime filtresi MC p>0.07 — DEAD" denmişti. Ama burada VOL_dec_low WR=83.3%, mc_p=0.0. Fark: Bu test `bl_cur()` (dar live gate, btc7d<0) üzerinde; Mega testi daha geniş universe üzerindeydi. Sonuç: **vol_decile 1-3 filtresi CURRENT GATE üzerinde anlamlı.**

2. **rv5m_high = low vol_decile:** Aynı 18 event. rv5m (realized volatility) yüksek ama vol_decile (background vol percentile) düşük. Bu = cascade sırasında vol patlıyor ama background vol düşük = compression → explosion pattern = en güçlü signal.

3. **Echo 45-120m en güçlü pencere:** 30-90m yerine 45-120m biraz daha geniş ve WR=94.1% ile en iyi. 17 event, OR gate'le de aynı.

4. **prebuildup > 0 critical:** EX_any_pre vs EX_no_pre: 78.9% vs 66.7%. Anchor öncesi herhangi bir cascade varsa significantly better. Mevcut live gate bunu filtreler mi? Hayır — eklenebilir.

5. **SHORT hold time:** BTC1M ile 2h → 3h → 4h tutmak avg artırır. 4h hold = +109.4bps vs 2h = +98.1bps. Küçük fark ama 4h daha iyi.

6. **SHORT score3 relaxation:** score>=3 SHORT için WR=70.8%, avg=+119.1bps, N=24, mc_p=0.0 vs score>=4: WR=76.9%, N=13. score3 daha düşük WR ama mc_p=0.0 ve N=24 → güçlü research kandidat.

### LONG Gate Karşılaştırması (kesin sonuç)

| Gate | N | WR | Avg bps | mc_p |
|---|---:|---:|---:|---:|
| btc7d<0 (current) | 31 | 74.2% | +81.7 | 0.001 |
| btc4h<0 only | 63 | 73.0% | +50.8 | 0.0 |
| btc3d<0 | 32 | 75.0% | +86.2 | 0.0 |
| btc4h<0 OR btc7d<0 | 68 | 73.5% | +55.0 | 0.0 |
| btc4h<0 added only (btc7d>=0) | 37 | 73.0% | +32.6 | 0.009 |

Sonuç: btc4h-only events (btc7d>=0) WR tutarlı ama avg düşük (+32.6). OR gate frekansı 31→68 yaparken avg 81→55. **btc3d<0 en iyi kalite/frekans dengesi** (btc7d kadar yüksek avg ama 1 event fazla).

### Actionable Sonraki Adımlar

| Öncelik | Aksiyon | Veri |
|---|---|---|
| 1 | **Echo 45-120m route shadow runner'a ekle** | N=17, WR=94.1%, mc_p=0.0 — güçlü |
| 2 | **SHORT BTC1M hold 2h → 4h güncelle** | +11.3bps avg artış, aynı N |
| 3 | **Vol decile 1-3 filtresi araştır** | Küçük N (18) — shadow bucket olarak izle |
| 4 | **prebuildup>0 gate araştır** | N=19 vs N=12 fark anlamlı — narrow universe gauntlet gerekli |
| 5 | **SHORT score3 gauntlet** | N=24, mc_p=0.0 — promotion öncesi holdout gerekli |

---

## 18. Shadow Route Implementation + SHORT Score3 Gauntlet (2026-07-01)

Yapılan değişiklikler:
- `tools/s34_realtime_shadow_runner.py`
  - `LONG_ECHO_45_120_SILENCE` shadow-only route eklendi.
    - Koşul: current LONG gate + önceki 45-120dk içinde ETH SELL bucketed cascade >=200K.
    - Davranış: anchor fiyatı T0 entry olarak saklanır; 30dk silence confirm olursa 4h hold ile kapanır. Noisy/follow-on gelirse `EXPIRED_NOISY`, PnL bucket'a girmez.
  - `SHORT_BTC1M_H4` shadow-only route eklendi.
    - Koşul: current SHORT prefilter + BTC SELL >=1M confirm, T+5dk sonrası.
    - Hold: 4h.
  - Bunlar observation-only; exchange order göndermez.
- `tools/s34_live_chart.py`
  - Shadow/Paper candidate bucket listesine iki yeni bucket eklendi:
    - `C_echo_45_120_silence`
    - `C_short_btc1m_h4`
- `tools/research_s34_short_score3_gauntlet.py`
  - SHORT score>=3/4, BTC 1M/2M, delay 5/10, hold 2h/3h/4h gauntlet scripti eklendi.

Doğrulama:
- `python -m py_compile tools\s34_realtime_shadow_runner.py tools\research_s34_short_score3_gauntlet.py` geçti.
- `python -m py_compile tools\s34_live_chart.py` geçti.
- `s34_state_machine_shadow_runner` restart edildi, alive PID `10808`.
- `s34_live_chart` restart edildi, alive PID `7044`.
- Live executor değişmedi: PID `19112`, active null, orders 0.
- Dashboard API yeni bucketları döndürüyor:
  - `C_echo_45_120_silence`
  - `C_short_btc1m_h4`

Rapor:
- `reports/research/s34/S34_SHORT_SCORE3_GAUNTLET.md`
- `reports/research/s34/S34_SHORT_SCORE3_GAUNTLET.json`

### SHORT score>=3 gauntlet sonucu
| Candidate | N | WR | Avg bps | T3R | Holdout Avg | Holdout T3R | Worst | TailN | Readiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S_score4_btc1m_delay10_hold180m | 10 | 90.0% | +145.9 | +631.8 | +240.1 | +720.4 | -34.5 | 0 | PAPER_CANDIDATE |
| S_score4_btc1m_delay10_hold120m | 10 | 90.0% | +133.2 | +523.6 | +219.6 | +658.9 | -11.2 | 0 | PAPER_CANDIDATE |
| S_score3_btc1m_delay5_hold120m | 19 | 78.9% | +108.0 | +1130.6 | +100.9 | -203.5 | -140.1 | 1 | RESEARCH_ONLY_HOLDOUT_WEAK |
| S_score3_btc1m_delay10_hold180m | 14 | 85.7% | +131.3 | +1011.1 | +172.6 | -30.2 | -34.5 | 0 | RESEARCH_ONLY_HOLDOUT_WEAK |
| S_score3_btc1m_delay10_hold240m | 14 | 78.6% | +166.1 | +866.6 | +217.6 | +10.9 | -164.7 | 1 | SHADOW_ONLY_TAIL |

Karar:
- `score>=3` SHORT genişletmesi overall iyi görünse de holdout T3R zayıf/negatif veya tail var. Live promotion yok.
- En temiz SHORT sonucu `score>=4 + BTC>=1M + delay>=10m + hold180m`; shadow/paper aday, live için ayrıca operator sign-off gerekir.
- Echo 45-120m ve SHORT BTC1M H4 artık dashboard/shadow üzerinden forward izlenecek.

---

## 19. Additional Shadow/Paper Candidate Routes (2026-07-01)

Operator request:
- Add `SHORT score4 BTC1M delay10 hold180`.
- Add `Prebuild-up >=2 / DOUBLE_CASCADE`.
- Add `OFI Silence + Buyers`.

Scope:
- Observation/shadow-only.
- No live executor, order logic, leverage, sizing, `.env`, `execution/`, `risk/`, or `brain/` changes.

Implementation:
- `tools/s34_realtime_shadow_runner.py`
  - Added `SHORT_BTC1M_D10_H3`.
    - Condition: current SHORT eligible universe, BTC SELL liquidation max >= $1M after T+10m.
    - Entry: paper SHORT at BTC confirmation time.
    - Hold: 3h.
  - Added `LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE`.
    - Condition: current LONG eligible universe, `prebuildup >= 2`.
    - Counts only if 30m silence confirms; noisy/follow-on expires without PnL.
    - Hold: 4h from anchor.
  - Added `LONG_OFI_SILENCE_BUYERS`.
    - Condition: current LONG eligible universe.
    - Counts only if 30m silence confirms and first 15m ETH agg-trade OFI is buyer-positive.
    - Hold: 4h from anchor.
- `tools/s34_live_chart.py`
  - Added candidate buckets:
    - `C_short_btc1m_d10_h3`
    - `C_double_cascade_prebuild2`
    - `C_ofi_silence_buyers`

Verification:
- `python -m py_compile tools\s34_realtime_shadow_runner.py tools\s34_live_chart.py` passed.
- `python tools\s34_realtime_shadow_runner.py --once --db data\microstructure.db` passed.
- `tools/s34_live_chart.py` AST parse passed.
- Restarted observation processes only:
  - `s34_state_machine_shadow_runner` alive PID `18800`.
  - `s34_live_chart` alive PID `3712`.
- Live executor unchanged:
  - `s34_state_machine_live_executor` alive PID `19112`.
  - `active` null, `orders_count` 0 at verification.
- Dashboard API `/api/data` returns all new buckets. Initial stats are N=0 because they are forward shadow buckets and have not closed trades yet.

Research context:
- `SHORT_BTC1M_D10_H3` was the cleanest SHORT gauntlet candidate:
  - N=10, WR 90.0%, avg +145.9 bps, T3R +631.8, holdout avg +240.1, holdout T3R +720.4, worst -34.5, tailN 0.
- `DOUBLE_CASCADE / prebuild>=2` and `OFI Silence + Buyers` are mechanism-consistent but smaller-N; they must remain shadow/paper until forward evidence accumulates.

---

## 20. State Machine Gate Cleanup (2026-07-01)

Operator-requested cleanup:
- Remove the disabled `if False and ...` guard from `SHORT_NOISY` in the shadow runner.
- Replace the old narrow LONG regime gate `btc7d < 0` with the researched frequency gate `btc4h < 0 OR btc7d < 0`.

Implementation:
- `tools/s34_state_machine_live_executor.py`
  - `long_eligible` now requires:
    - `(btc4h_bps < 0 OR btc7d_bps < 0)`
    - all other active LONG gates unchanged: not bull pullback, not EUROPE, not US 13-14, not Mon/Wed, `sync_k < 200K`, `score_if_silence >= 3`.
- `tools/s34_realtime_shadow_runner.py`
  - Same LONG regime gate as live: `(btc4h < 0 OR btc7d < 0)`.
  - `SHORT_NOISY` shadow route re-enabled by removing `if False and`.
  - `SHORT_NOISY` remains shadow/paper only; no live order route was added.

Verification:
- `python -m py_compile tools\s34_state_machine_live_executor.py tools\s34_realtime_shadow_runner.py` passed.
- `python tools\s34_realtime_shadow_runner.py --once --db data\microstructure.db` passed.
- Restarted updated runtime processes:
  - `s34_state_machine_live_executor` alive PID `2492`.
  - `s34_state_machine_shadow_runner` alive PID `17076`.
- `status_eclipse.ps1` verification:
  - live mode `LIVE`
  - live rule `S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3`
  - active empty/null
  - live orders count `0`

Research context:
- `btc4h<0 OR btc7d<0` was the best tested LONG frequency relaxation:
  - N 31 -> 68 in backfill, ~15/month, WR ~73.5%, avg +55 bps.
  - It preserves WR while materially increasing frequency versus `btc7d<0` only.
- This is a live gate expansion, not a new alpha family.

---

## 21. BUY-Side State-Machine Symmetry Gauntlet (2026-07-01)

User request:
- Test symmetric family for ETH BUY liquidations.
- Clarify semantics:
  - `ETH SELL liquidation` = long liquidation / forced sell.
  - `ETH BUY liquidation` = short liquidation / forced buy.
  - BUY-side expected directions:
    - `ETH BUY -> SHORT` = mean-reversion / fade after short squeeze.
    - `ETH BUY -> LONG` = continuation after short squeeze.

Scope:
- Offline research only.
- No live executor, `.env`, order logic, leverage, sizing, or dashboard files changed.

New script:
- `tools/research_s34_buy_side_state_machine_gauntlet.py`

Reports:
- `reports/research/s34/S34_BUY_SIDE_STATE_MACHINE_GAUNTLET.md`
- `reports/research/s34/S34_BUY_SIDE_STATE_MACHINE_GAUNTLET.json`

Method:
- Reconstructed knowable ETH BUY 200K anchors directly from `liquidations`.
- Tested 76 cells:
  - BTC BUY confirm thresholds: 500K / 1M / 2M.
  - Delay: 5m / 10m.
  - Hold: 1h / 2h / 3h / 4h.
  - States: silence / noisy / same-side propagation / BTC confirmation.
  - Echo 45-120m.
  - Prebuild-up >=2 / double cascade.
  - sync_k, btc4h/btc7d mirrored regime, DOW/session diagnostics.
  - Top-3 winner removed, chronological holdout, walk-forward folds, no-overlap execution, max-stat permutation.

Dataset:
- ETH BUY 200K knowable anchors: `563`.
- Range: `2026-02-15T22:47:14.217000+00:00` -> `2026-07-01T04:45:52.552000+00:00`.

Key results:

| Family | Best candidate | N | WR | Avg bps | T3R bps | Readiness |
|---|---|---:|---:|---:|---:|---|
| BUY -> LONG continuation | `C_score4_btc2000k_delay10_long_h120` | 4 | 50.0% | -27.1 | -116.7 | LOW_N_RESEARCH_ONLY |
| BUY -> LONG same-side propagation | `C_same_side_follow_long_h60` | 142 | 37.3% | -9.3 | -2176.7 | REJECT_T3R |
| BUY -> SHORT mean-reversion | `F_silence_short_h60` | 184 | 67.9% | +22.7 | +3426.3 | SHADOW_ONLY_TAIL |

Top BUY->SHORT fade cells:
- `F_silence_short_h60`: N 184, WR 67.9%, avg +22.7, sum +4183.1, T3R +3426.3, holdout avg +20.4, holdout T3R +638.4, no-overlap T3R +3140.4, folds 5/5 positive, worst -334.3, tail100 N=5.
- `F_silence_short_h120`: N 184, WR 63.6%, avg +18.0, T3R +2380.4, holdout T3R +214.3, no-overlap T3R +2264.5, worst -548.1, tail100 N=16.
- `F_prebuild2_silence_short_h60`: N 54, WR 70.4%, avg +33.4, T3R +1286.8, holdout avg +41.9, holdout T3R +388.0, no-overlap T3R +1286.8, worst -206.6, tail100 N=2.

Permutation:
- Max-stat permutation over searched cells:
  - observed max T3R `3426.3`
  - null p95 max T3R `2197.2`
  - p_right `0.01`
- Interpretation: there is likely a real BUY-side SHORT-fade effect after multiple-comparison correction, but tail risk prevents paper/live promotion.

Direct answers:
- ETH BUY -> LONG continuation alpha: **No**. BTC-confirm continuation cells are low-N and/or negative; same-side propagation LONG is outright rejected.
- ETH BUY -> SHORT mean-reversion alpha: **Possible shadow lead**. Silence-based 1h SHORT fade clears permutation and no-overlap, but has large tails, so it is not paper/live ready.
- New trade frequency: potentially yes as a **separate shadow-only BUY-side fade family**, especially `BUY silence -> SHORT h60`, but it needs tail management and forward shadow.
- Relationship to ETH SELL family: not just the same route inverted. SELL-side current live is ETH SELL -> LONG/SHORT state machine; BUY-side viable lead is specifically `BUY + silence -> SHORT fade`, while BUY continuation is dead.

Next research if pursued:
1. Tail anatomy for `F_silence_short_h60`: identify whether the -334/-548 bps tails are clustered by day/session/BTC regime.
2. Protective stop / time-stop sweep for BUY->SHORT h60 only.
3. Forward shadow route only after tail controls; no live promotion from this result alone.

---

## 22. BUY-Side Fade Follow-Up Tests (2026-07-01)

User request:
- Run all proposed tests for the BUY-side SHORT fade lead.
- Focus: tail anatomy, stop/time-stop, exit shape, silence/confirmation cost, SELL-family overlap, ask-depth mirror, cross-asset resonance, navigation labels.

Scope:
- Offline research only.
- No live executor, shadow runner, `.env`, order logic, leverage, sizing, or dashboard files changed.

New script:
- `tools/research_s34_buy_side_fade_followup.py`

Reports:
- `reports/research/s34/S34_BUY_SIDE_FADE_FOLLOWUP.md`
- `reports/research/s34/S34_BUY_SIDE_FADE_FOLLOWUP.json`

Baseline:
- `F_silence_short_h60`: N 184, WR 67.9%, avg +22.7 bps, sum +4183.1, T3R +3426.3, worst -334.3, tail100 N=5.

### Key results

#### Tail anatomy
- Tail subset: N 5, avg -187.0, worst -334.3.
- Non-tail subset: N 179, WR 69.8%, avg +28.6, T3R +4361.2, worst -86.6.
- US session carries most tails:
  - US: N 108, tail100 N=4.
  - ASIA: N 55, tail100 N=0.
- DOW weakness:
  - Monday: N 36, avg +9.0, T3R +5.5, tail100 N=2.
  - Tuesday: N 35, avg +13.4, T3R +33.1, tail100 N=2.
- Echo is not safe here:
  - echo_45_120: N 77, tail100 N=5.
  - no_echo: N 107, tail100 N=0.

#### Stop / time-stop sweep
Best variant by T3R:
- `h45_sl75`: N 184, WR 67.9%, avg +24.8, sum +4567.7, T3R +3918.1, worst -80.0, tail100 N=0, SL exits 12.

Other strong variants:
- `h45_sl50`: avg +24.6, T3R +3878.6, worst -55.0, tail100 N=0.
- `h45_slnone`: avg +24.6, T3R +3871.4, worst -255.8, tail100 N=2.
- `h30_slnone`: WR 69.6%, avg +23.9, T3R +3791.1, worst -101.6, tail100 N=1.

Conclusion:
- The original 60m hold is not optimal.
- 45m hold is better.
- 75bps stop appears to remove all -100bps tails while improving T3R, but this is still backtest-only and needs forward shadow.

#### Exit shape
- h20: WR 76.6%, avg +19.8, worst -96.7, tail100 N=0.
- h30: WR 69.6%, avg +23.9, T3R +3791.1.
- h45: WR 68.5%, avg +24.6, T3R +3871.4.
- h60: WR 67.9%, avg +22.7, T3R +3426.3.
- h90/h120 degrade tails.

Conclusion:
- The edge resolves mostly in 20-45 minutes.
- Holding past 60m worsens tail risk.

#### Silence / confirmation cost
- T0 entry with later silence label works.
- Waiting for confirmation kills the edge:
  - `silence30_t0_short_h60`: N 184, WR 67.9%, avg +22.7.
  - `silence30_confirmed_entry_h60`: N 184, WR 39.7%, avg -10.2.
- Same pattern for 10/15/20/45m confirmation entries.

Conclusion:
- This is a T0 entry / post-classified state lead. Waiting for silence confirmation is not tradeable as an entry improvement.
- If used, it must be shadowed as "enter at anchor, monitor state"; confirmation is diagnostic/management, not entry trigger.

#### SELL live-family overlap
- Approx SELL-family event count: 109.
- BUY fade overlap with SELL-family is low:
  - overlap_30m rate 4.3%.
  - overlap_60m rate 7.6%.
  - overlap_120m rate 17.4%.
- No-overlap remains positive:
  - 60m no-overlap: N 170, WR 67.6%, avg +23.0, T3R +3238.8.

Conclusion:
- BUY-side fade likely adds independent frequency rather than simply duplicating current SELL-family events.

#### Ask-depth / absorption mirror
- Ask depth >=100K: N 55, WR 70.9%, avg +25.2, worst -86.6, tail100 N=0.
- Ask depth <50K: N 120, WR 68.3%, avg +24.3, worst -334.3, tail100 N=4.
- Ask imbalance: N 38, WR 71.1%, avg +27.4, worst -86.6, tail100 N=0.

Conclusion:
- Ask-side absorption is a plausible tail-control/navigation feature for BUY->SHORT fade.

#### Cross-asset BUY resonance
- sync_ge500k: N 29, WR 82.8%, avg +30.3, worst -43.6, tail100 N=0.
- sync_lt200k: N 124, WR 63.7%, avg +18.0, worst -334.3, tail100 N=5.

Conclusion:
- Contrary to SELL-side intuition, BUY-side high sync is not a danger; it improves fade quality and cuts tails.
- `BUY_SYNC_HIGH_TAIL_WARNING` as initially named is wrong for silence-fade subset; high sync is more like `BUY_RESONANT_EXHAUSTION_OK`.

#### Navigation labels
- `BUY_CONTINUATION_DANGER`: N 316, WR 39.9%, avg -26.1, T3R -8936.8, tail100 N=61.
- `BUY_SILENCE_FADE_WATCH`: N 247, WR 64.4%, avg +18.4, T3R +3596.1.
- `BUY_PREBUILD2_FADE_WATCH`: N 69, WR 66.7%, avg +31.1, T3R +1406.4.
- `BUY_SYNC_HIGH_TAIL_WARNING`: N 209, WR 56.5%, avg -9.4 overall, but in the silence-fade subset high sync is positive. Label needs context.

### Decision
- BUY-side SHORT fade is stronger after follow-up than initial gauntlet suggested.
- Best offline candidate:
  - `ETH BUY 200K + silence-state + SHORT at T0 + 45m hold + 75bps SL`.
  - Stats: N 184, WR 67.9%, avg +24.8, sum +4567.7, T3R +3918.1, worst -80.0, tail100 N=0.
- Still not live-ready because silence is post-classified and stop simulation is idealized mark-path. Next valid step is shadow bucket, not live.

Next recommended tests:
1. Build shadow bucket for BUY-side T0 SHORT with h45/sl75 observer.
2. Execution realism for stop fill: mark-path stop may understate gap-through.
3. Forward shadow validation with labels:
   - `BUY_SILENCE_FADE_WATCH`
   - `BUY_PREBUILD2_FADE_WATCH`
   - `BUY_RESONANT_EXHAUSTION_OK`
   - `BUY_CONTINUATION_DANGER`

---

## 16. Kapsamlı Araştırma Fikirleri — V2 (2026-07-01)

Rapor: `reports/research/s34/S34_RESEARCH_IDEAS_V2.md`

### 3 Onaylanmış Sinyalin Derinleştirilmesi

| Sinyal | /ay | WR | En Kritik Açık Sorular |
|---|---:|---:|---|
| LONG gate: btc4h OR btc7d | 15 | 73.5% | btc4h tek başına driver mı? Session × gate interaksiyon? Mon block OR'da geçerli mi? |
| SHORT_NOISY BTC>=1M delay5 | 5.5 | 76% | Optimal hold (2h/3h?), BTC 500K threshold stable mi?, vol filter katkısı? |
| Echo cascade 30-90m silence | 16.8 | 68.4% | Window optimize (20-60/30-90/45-120?), echo büyüklük oranı, 3. cascade degradasyon? |

### 10 Yeni Araştırma Alanı (öncelik sırasıyla)

| ⭐ | Alan | Temel Hipotez | Veri Kaynağı | Öncelik |
|---|---|---|---|---|
| ⭐⭐⭐⭐⭐ | State Transition (2B) | SILENCE/NOISY/BTC sequence → 4 partition, WR çok farklılaşır | `liquidations` | P1 |
| ⭐⭐⭐⭐⭐ | BTC Lead-Lag (2D) | ETH anchor'da ETH-BTC timing delta → ETH-önce = daha iyi LONG | `liquidations` BTC+ETH | P2 |
| ⭐⭐⭐⭐ | Liquidity Vacuum OFI (2G) | Silence sırasında `agg_trades` net OFI>0 → alıcılar dolduruyor | `agg_trades` (indexed) | P3 |
| ⭐⭐⭐ | Vol Compression (2J) | Anchor öncesi vol düşük → cascade sürpriz → bounce güçlü | `vol_state` | P4 |
| ⭐⭐⭐⭐⭐ | Second Wave (2C) | İlk cascade sonrası 90-180m içinde 2. büyük cascade = 2. dalga → LONG | `liquidations` | P7 |
| ⭐⭐⭐⭐⭐ | Simetri (2A) | SHORT simetriği: DOW/score SHORT için farklı davranıyor mu? | `liquidations`, `mark_prices` | P8 |
| ⭐⭐⭐⭐ | Cascade Density (2F) | density_24h=0 (temiz piyasa) = en iyi LONG; density>=3 = kötü | `liquidations` | P9 |
| ⭐⭐⭐⭐ | Cascade Exhaustion (2H) | Azalan notional seri: prebuildup cascades küçülüyor → LONG güçlü | `liquidations` | P10 |
| ⭐⭐⭐⭐ | Failed Bounce → SHORT (2E) | price(+5m) > price(anchor) + noisy follow → SHORT | `mark_prices`, shadow ledger | P5-alt |
| ⭐⭐⭐ | Cluster Shape (2I) | cascade_rise_time_sec, fingerprint_class → spike vs gradual | `detector_signals` (73), `agg_trades` | Düşük |

### En Hızlı Çalışabilir Testler (mevcut veriden, yeni tablo gerektirmez)
1. **State partition (2B-T1):** `liquidations` tablo var, 30 satır Python → 1 session
2. **BTC timing delta (2D-T1):** `liquidations` BTC + ETH var → 1 session
3. **Density quartile (2F-T1):** `liquidations` groupby anchor_ts → hızlı
4. **Vol expansion at cascade (2J-T2):** `vol_state` nearest join → 1 session

### `agg_trades` OFI için Sorgu Şablonu (2G-T5)
```python
# anchor'dan sonra 10dk içinde OFI hesapla (indexed sorgu)
sql = """
    SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END) as buy_notional,
           SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) as sell_notional
    FROM agg_trades
    WHERE symbol='ETHUSDT' AND ts_ms BETWEEN ? AND ?
"""
# OFI = buy_notional - sell_notional
```
`idx_trade_symbol_ts (symbol, ts_ms)` index → hızlı

---

## 17. BUY-Side Fade Shadow Bucket Implementation (2026-07-01)

Scope:
- Observation/shadow only.
- Live executor untouched.
- No `.env`, order logic, leverage, sizing, `execution/`, `risk/`, or `brain/` changes.

Implemented files:
- `tools/s34_realtime_shadow_runner.py`
- `tools/s34_live_chart.py`

New shadow-only route:
- `BUY_FADE_SHORT_H45_SL75`
- Trigger: fresh ETH BUY liquidation anchor at the state-machine threshold.
- Direction: paper SHORT.
- Exit model:
  - 45 minute time exit.
  - 75 bps paper stop-loss observer.
- State labels:
  - `PENDING` after entry.
  - `NOISY` if a follow-on ETH BUY cascade >=50K arrives from T+60s to T+30m.
  - `SILENCE` if no follow-on cascade arrives by T+30m.
- Entry filters:
  - Excludes EUROPE session.
  - Excludes bear-squeeze continuation risk: `eth1h < -20 bps and btc4h < -50 bps`.

Dashboard buckets added:
- `C_buy_fade_h45_sl75_all`
- `C_buy_fade_h45_sl75_silence`

Verification:
- `python -m py_compile tools\s34_realtime_shadow_runner.py tools\s34_live_chart.py` passed.
- `python tools\s34_realtime_shadow_runner.py --once --db data\microstructure.db` passed.
- Restarted shadow runner and chart only.
- Live executor was not restarted or changed.
- Post-check:
  - live executor alive, PID 2492.
  - live active blank/null.
  - live orders 0.
  - shadow runner alive, PID 11916.
  - chart alive, PID 10600.
  - dashboard `/api/data` exposes both new BUY fade buckets.

Current forward status:
- New BUY fade buckets start at N=0 until future BUY-side shadow trades close.
- This route is not live-ready; it is a forward observer for the offline candidate:
  - `ETH BUY 200K + T0 SHORT + 45m hold + 75bps SL`
  - offline follow-up: N 184, WR 67.9%, avg +24.8 bps, T3R +3918.1, tail100 N=0.

---

## 18. LONG Relax + Management Suite (2026-07-01)

Report:
- `reports/research/s34/S34_LONG_RELAX_MANAGEMENT_SUITE.md`
- `reports/research/s34/S34_LONG_RELAX_MANAGEMENT_SUITE.json`

Scope:
- Research-only.
- Tested LONG gate relax, tail detector, dynamic hold, confidence score, exit-by-state, route fusion, adaptive stop, multi-stage entry, and sizing.
- No live executor, `.env`, leverage, sizing, order logic, or runtime state changes.

Script:
- `tools/research_s34_long_relax_management_suite.py`

Verification:
- `python -m py_compile tools\research_s34_long_relax_management_suite.py` passed.
- Full suite completed and wrote MD/JSON reports.

### Baselines
- `long_current_relax_or` (`btc4h < 0 OR btc7d < 0`): N 58, WR 65.5%, avg +39.6 bps, sum +2297.5, T3R +1341.9, worst -305.3, tail100 7, ~13.3/month.
- `long_strict_btc7d`: N 28, WR 71.4%, avg +72.5 bps, sum +2029.1, T3R +1073.4, worst -305.3, tail100 3, ~6.4/month.
- `short_current_btc2m_d5_h2`: N 7, WR 85.7%, avg +99.0 bps, sum +693.2, T3R -58.4, worst -169.6, tail100 1.
- `short_btc1m_d5_h4`: N 12, WR 66.7%, avg +150.0 bps, sum +1800.4, T3R +327.1, worst -165.8, tail100 3.

### Important Findings
- Current relaxed LONG increases frequency to ~13.3/month, but quality falls below target: WR 65.5%, avg +39.6.
- Strict btc7d-only is cleaner but lower frequency: ~6.4/month, WR 71.4%, avg +72.5.
- Sync relaxation to 300K/500K increases frequency but adds tails; not a clean live upgrade.
- Confidence >=5 improves WR to 74.3% but avg remains +49.8 and worst tail remains -305.3.
- Silence-only state is very strong but post-classified: N 19, WR 84.2%, avg +102.5, tail100 0. It is useful for navigation/management, not an entry gate at T0.
- NOISY state is the problem: N 39, WR 56.4%, avg +9.0, T3R -296.6, tail100 7.
- Exiting on noisy follow cuts tail100 to 0, but also cuts expectancy: avg +15.8 vs +39.6. This is defensive management, not an alpha improvement.
- Reversing SHORT on noisy follow failed: avg -5.6, tail100 13.
- Dynamic hold:
  - LONG 4h remains the best balanced hold; 3h is close but weaker.
  - 8h worsens tails and WR.
  - SHORT sample is too small; current 2h still better than 4h on the current BTC2M route.
- Adaptive stops reduce tail but also reduce expectancy:
  - LONG SL75: tail100 0, avg +17.7.
  - LONG SL150: avg +30.4, tail100 11.
  - No stop variant beats baseline on expectancy.
- Multi-stage entry:
  - T+5m is close to baseline and slightly improves hold split, but not enough.
  - T+15 after bounce has tail100 0 and avg +54.2 but WR 68.8%, N 32, below promotion threshold.
- Position sizing:
  - Confidence scaling increases scaled sum but increases worst scaled loss.
  - Half-size tail-risk reduces worst trade but also reduces expectancy.

### Decision
- `NO_LIVE_CHANGE`.
- Promotion rule was: N >= 30, WR >= 70%, avg > 70 bps, T3R > 0.
- No candidate passed the full promotion rule while also improving trade frequency.
- Current live remains unchanged.

### Next Shadow/Research Leads
1. Track `T+15_AFTER_BOUNCE` as a shadow/navigation observer, not live: N 32, WR 68.8%, avg +54.2, tail100 0.
2. Track `NOISY_EXIT_DEFENSE` as a defensive observer: tail100 0 but avg only +15.8.
3. Keep silence confirmation as a dashboard quality label: strong state, but not knowable at T0.

---

## 19. Final Live Update: T+15 Bounce LONG (2026-07-01)

Operator instruction:
- Promote the best liveable version after comprehensive management/tail tests.
- Open bucket and live route.

Files changed:
- `tools/s34_state_machine_live_executor.py`
- `tools/s34_realtime_shadow_runner.py`
- `tools/s34_live_chart.py`
- `tools/research_s34_long_relax_management_suite.py`

Final live route:
- Rule remains allowed as `S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3` to avoid `.env` changes.
- LONG leg changed from T0 provisional entry to:
  - `LONG_T15_BOUNCE_CONFIRM`
  - Queue anchor at ETH SELL >=200K if live gates pass.
  - Wait until anchor +15 minutes.
  - Open LONG only if ETH mark at T+15m is above anchor mark.
  - Exit remains anchor +4h.
  - BTC>=2M delay5 SHORT confirmation still replaces the LONG if it appears before/after the LONG opens.
- SHORT leg unchanged:
  - BTC SELL >=2M confirmation after >=5m inside 30m window.
  - 2h hold.
  - SHORT replacement logic remains.

Why this version:
- Full management suite found no candidate meeting the strict promotion rule, but operator required live promotion.
- Among executable candidates, `T+15 after bounce` was the cleanest tail-control candidate:
  - N 32, WR 68.8%, avg +54.2 bps, T3R +970.4, worst -85.3, tail100 0, ~7.4/month.
- Compared with current relaxed T0 LONG:
  - T0 relaxed: N 58, WR 65.5%, avg +39.6, worst -305.3, tail100 7, ~13.3/month.
  - T+15 bounce reduces frequency but materially cuts observed tail.

Shadow/dashboard:
- Shadow runner now mirrors the live LONG route as `LONG_T15_BOUNCE`.
- Dashboard candidate bucket added:
  - `LIVE_long_t15_bounce`
- Existing research-only bucket remains:
  - `C_buy_fade_h45_sl75_all`

Verification:
- `python -m py_compile tools\s34_state_machine_live_executor.py tools\s34_realtime_shadow_runner.py tools\s34_live_chart.py` passed.
- `python tools\s34_realtime_shadow_runner.py --once --db data\microstructure.db` passed.
- Live executor dry-run `--once` passed after exchange metadata access.
- Clean `start_eclipse.ps1` restart completed.
- Post-restart status:
  - `s34_state_machine_live_executor` alive, PID 19824.
  - live mode `LIVE`.
  - live `allowed=true`.
  - live active `null`.
  - live open orders count `0`.
  - pending count `0`.
  - shadow runner alive, PID 15316.
  - live chart alive, PID 10572.
  - collector/bookticker/heartbeat alive.
  - watchdog overall `GREEN`.
  - dashboard `/api/data` returns candidate buckets:
    - `LIVE_long_t15_bounce`
    - `C_buy_fade_h45_sl75_all`

Current state:
- Final live is armed and will trade the updated route if a valid signal arrives.
- No open position and no open state-machine orders at the time of verification.

---

## 23. Echo Expansion — Frekans Artırma + Tail/WR (2026-07-01)

Rapor: `reports/research/s34/S34_ECHO_EXPANSION.json` + `.md`
Script: `tools/research_s34_echo_expansion.py`
Evren: 597 ETH SELL 200K anchor, 4.52 ay, FEE=5bps, hold=4h. Tek process, read-only DB.

**Baz:** "Echo geniş" = echo_30_90 + silence, hiç gate yok. Birebir üretildi:
`M0_user_baseline`: N=76, 16.8/ay, WR=68.4%, avg=+63.1, worst=-237, tail=4, mc_p=0.0. ✓

### Ana Sonuç: Frekans ↔ Tail-güvenliği TERS ORANTILI

Echo ailesinde 17/ay üstüne çıkmak mümkün ama WR düşer ve tail büyür. 17/ay üstü frekans, sadece down-regime'de bile kaybeden zayıf echo'lardan gelir; rejim gate bu geniş pencerede tail'i temizleyemiyor.

**Frekans yönü (17/ay üstü):**
| Config | /ay | WR | avg | worst | tail | mc_p |
|---|---:|---:|---:|---:|---:|---:|
| M_union_raw (e20-60∪30-90∪45-120∪60-180) | 25.2 | 65.8% | +40.1 | -464 | 12 | 0.0 |
| M_e20_150_raw | 21.9 | 65.7% | +44.7 | -464 | 9 | 0.0 |
| M_e30_150_raw | 20.8 | 66.0% | +47.3 | -464 | 8 | 0.0 |
| M_union_reg_s150 (rejim + stop150) | 20.6 | 66.7% | +46.4 | **-155** | 10 | 0.0 |
| M_e30_150_reg_s150 | 17.2 | 67.9% | +55.4 | **-155** | 7 | 0.0 |

> Frekansı 20-25/ay'a çıkarmak MÜMKÜN ama WR 66%'a düşer, avg +40-47'ye iner. Rejim geniş pencerede tail'i SİLMİYOR (sadece stop150 worst'u -155'e kırpar). Yani frekans, kalite pahasına satın alınıyor — bedavaya değil.

### Asıl Kazanç: TERS yön — Tail/WR (rejim gate tight echo'da)

Aynı rejim gate DAR echo penceresinde tail'i SIFIRA indiriyor + WR'yi fırlatıyor:
| Config | /ay | WR | avg | worst | tail | mc_p | wf |
|---|---:|---:|---:|---:|---:|---:|---:|
| echo_30_90 + gate (F1) | 8.4 | 71.1% | +71.6 | -197 | 2 | 0.0 | 4/5 |
| **echo_30_90 + regime (S1)** | 7.1 | **81.2%** | +92.5 | **-85.9** | **0** | 0.0 | 5/5 |
| **echo_30_90 + regime + prebuild>0 (S4)** | 5.3 | **91.7%** | +117.8 | **-22.9** | **0** | 0.0 | 5/5 |
| echo_30_90 count>=2 (double echo) | 1.8 | **100%** | +142.4 | +35.3 | 0 | 0.003 | 5/5 |
| echo_30_120 count>=2 | 3.3 | **100%** | +127.8 | +35.3 | 0 | 0.001 | 5/5 |

> `regime = btc4h<0 OR btc7d<0`. Bu tek filtre echo bazının tail'ini (-237/-464) tamamen kaldırıyor ve WR'yi 68→81'e taşıyor. +prebuildup>0 → WR 91.7%, worst -22.9. Double-echo (2+ prior cascade) → WR 100% ama nadir.

### En İyi Frekans+Kalite Dengesi (orta yol)

| Config | /ay | WR | avg | worst | tail | mc_p | wf | no-overlap /ay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **C_echo30_120_regime** | 9.7 | 75.0% | +65.2 | -464 | 2 | 0.001 | 5/5 | 7.7 (WR 68.6) |
| E_union_regime | 10.0 | 73.3% | +62.8 | -464 | 2 | 0.001 | 5/5 | 8.0 (WR 66.7) |
| D_echo30_120_reg_stop100 | 9.7 | 70.5% | +62.1 | **-105** | 7 | 0.001 | 4/5 | 7.7 |

> Pencereyi 90→120dk açıp rejim eklemek: gated e30_90'a göre hem daha çok frekans (9.7 vs 8.4) hem daha yüksek WR (75 vs 71). Temiz upgrade.

### Stop Simülasyonu (path-min gerçek stop-out)

| Stop | echo baz WR | avg | worst | tail | echo+regime WR | avg | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 71.1% | +71.6 | -197 | 2 | 81.2% | +92.5 | -85.9 |
| -200 | 71.1% | +71.4 | -205 | 2 | 81.2% | +92.5 | -85.9 |
| **-150** | 71.1% | **+72.7** | **-155** | 2 | 81.2% | +92.5 | -85.9 |
| -100 | 68.4% | +66.2 | -105 | 5 | 78.1% | +83.1 | -105 |
| -75 | 60.5% | +55.3 | -80 | 0 | 71.9% | +75.9 | -80 |
| -50 | 52.6% | +48.5 | -55 | 0 | 62.5% | +67.0 | -55 |

> **-150 bps stop bedava tail kapağı:** WR değişmiyor, avg hafif artıyor (derin kaybı kırptığı için), worst -155'e sabitleniyor. -100 altındaki stoplar edge'i öldürüyor (recover eden trade'leri stop'luyor). Rejim gate zaten worst'u -85.9 yapıyor, yani rejim varken stop gereksiz.

### Karar / Öneriler (research-only — live'a dokunulmadı)

1. **Frekans hedefi net değilse doğru hamle YUKARI değil AŞAĞI:** rejim gate (btc4h<0 OR btc7d<0) echo'ya eklemek tek en büyük tail+WR kaldıracı. `echo_30_90 + regime` = 7/ay, WR 81%, tail 0.
2. **Frekans+kalite dengesi:** `echo_30_120 + regime` (9.7/ay, WR 75%, tail 2, wf 5/5) — mevcut gated e30_90'a temiz upgrade, shadow bucket adayı.
3. **17/ay üstü zorunluysa:** `union_reg_s150` (20.6/ay) veya `e30_150_reg_s150` (17.2/ay) + mutlaka -150 stop; ama WR 66-68'e razı ol, tail_n hâlâ 7-10.
4. **Stop politikası:** -150 bps herhangi echo route'una bedava eklenebilir. -100 ve altı yasak (edge kill).
5. Bir sonraki adım: `echo_30_120 + regime` ve `echo_30_90 + regime + prebuild` shadow bucket olarak forward izle; live promotion öncesi holdout + forward accumulation.

### Guardrail
Bu testler research-only. `tools/s34_state_machine_live_executor.py`, `.env`, order logic, leverage, sizing değiştirilmedi. Live executor PID/route dokunulmadı.

---

## 24. Echo Live-Readiness Gauntlet (2026-07-01)

Rapor: `reports/research/s34/S34_ECHO_LIVE_GAUNTLET.json` + `.md`
Script: `tools/research_s34_echo_live_gauntlet.py`
Amaç: `echo + regime` adayını canlıya almadan önce giriş mekanizması, hold, tail, maliyet, rejim stabilite testleri.

### En kritik bulgu: GİRİŞ mekanizması e30_90 vs e30_120'yi ayırıyor

Live executor T0 provisional yerine **T+15 bounce-confirm** kullanıyor. Echo edge'i öne-yüklü:
| Aday | T0 | T+15 confirm (live mekanizması) |
|---|---|---|
| echo_30_90 + regime | WR 81.2%, avg +92.5, tail 0, mc_p 0.0 | WR 69.0%, avg +57.6, tail 1, **mc_p 0.004** (hâlâ anlamlı) |
| echo_30_120 + regime | WR 75.0%, avg +65.2, tail 2 | WR 63.4%, avg +26.0, tail 3, **mc_p 0.114** (ÖLÜ) |

> **echo_30_120 canlı T+15 mekanizmasında ölüyor** (mc_p 0.114). Ekstra frekansı veren geniş-pencere trade'lerinin edge'i 15dk beklenince kayboluyor. **Canlı için doğru pencere echo_30_90.** Delay taraması: d5/d10 e30_90'da WR 75% (kabul edilebilir), e30_120'de bozuluyor.

### Tail adli analizi — 2 tail eventinin ikisi de yapısal olarak zaten bloklu

echo_30_120+regime'deki 2 tail (net<-100):
| Zaman | net | sess | btc7d | be_ratio | e3090 |
|---|---:|---|---:|---:|---:|
| 2026-02-27 13:36 | -120.6 | US | -175.8 | 0.39 | **False** |
| 2026-06-25 13:00 | -464.5 | US | -480.5 | **4.87** | **False** |

> **İkisi de US 13:00-14:00 UTC** → mevcut live LONG gate zaten US13-14 blokluyor. **İkisi de echo_30_90 dışında** (e3090=False). Yani `echo_30_90` kısıtı + US13-14 blok = **tail SIFIR**. Ayrıca `be_ratio<2` vetosu -464'ü siliyor (BTC cascade ETH'nin 4.87 katı — mega-research vetosu doğrulandı). Tek-veto sonuçları: `V_not_us1314` → tail 0 WR 85.3%; `V_be_lt2` → tail 1 WR 77.5%; `V_echo3090` → tail 0 WR 81.2%.

### Hold taraması — 6h, 4h'tan daha iyi (echo_30_90)
| Hold | WR | avg | worst |
|---|---:|---:|---:|
| 2h | 84.4% | +47.5 | -64.0 |
| 3h | 84.4% | +82.1 | -64.5 |
| 4h (mevcut) | 81.2% | +92.5 | -85.9 |
| **6h** | **84.4%** | **+104.6** | **-27.4** |

> echo_30_90 için 6h hold hem avg hem WR hem tail'i iyileştiriyor. Exit süresini 4h→6h uzatmak bedava upgrade (echo subset'inde).

### Maliyet duyarlılığı — echo_30_90 sağlam
`echo_30_90+regime` fee 5→15 bps: avg +92.5 → +82.5, WR sabit 81.2%, tail 0. **VIP-olmayan fee'de bile canlı.** (Karşıt: e30_120_t15 fee10'da avg +21, mc_p 0.161 — ölü.)
No-overlap (gerçekçi tek-pozisyon): echo_30_90 = 5.7/ay, WR 76.9%, avg +87.3, tail 0.

### Rejim bacağı
- `btc4h<0` frekansı taşıyan bacak (9.3/ay); `btc7d<0` tek başına 5.1/ay; OR = 9.7/ay en iyi denge.
- `btc4h_only` (btc7d≥0) subset: worst -33.7, tail 0, WR 71.4% — en temiz alt-küme.
- Ay-ay: her ay pozitif; 2026-04 zayıf (WR 50%, N=6) ama pozitif; 2026-06 en güçlü (WR 82.4%, avg +94.1).

### Live-Readiness Scorecard
| Aday | /ay | WR | avg | tail | mc_p | wf | no-ov /ay | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **echo_30_90 + regime (T0)** | 7.1 | 81.2% | +92.5 | 0 | 0.0 | 5/5 | 5.7 | **PAPER_CANDIDATE** |
| echo_30_90 + regime (T+15) | 6.4 | 69.0% | +57.6 | 1 | 0.004 | 4/5 | 5.3 | RESEARCH_ONLY |
| echo_30_120 + regime (T0) | 9.7 | 75.0% | +65.2 | 2 | 0.001 | 5/5 | 7.7 | SHADOW_ONLY |
| echo_30_120 + regime + stop150 | 9.7 | 75.0% | +71.4 | 2 | 0.0 | 5/5 | 7.7 | SHADOW_ONLY |
| echo_30_120 + prebuild>0 | 7.1 | 81.2% | +83.4 | 1 | 0.001 | 5/5 | 6.0 | SHADOW_ONLY |
| echo_30_120 + regime (T+15) | 9.1 | 63.4% | +26.0 | 3 | 0.114 | 4/5 | 7.3 | RESEARCH_ONLY |

### Canlı yol haritası (öneri — operator sign-off gerekli)

1. **Tek PAPER_CANDIDATE = `echo_30_90 + regime`.** Bu, mevcut live LONG gate'in (not bull / not EU / not US13-14 / not Mon-Wed / sync<200K / btc4h OR btc7d) üstüne **"önceki 30-90dk içinde ETH SELL 200K cascade (echo)"** kalite şartı eklenmiş hâli. Yani ayrı bir alfa değil, mevcut route'un yüksek-güven refinement'ı (WR ~73 → 81).
2. **Giriş:** echo subset'i T0'da temiz (tail 0, US13-14 zaten bloklu). Canlı T+15 bounce ise WR'yi 81→69'a düşürür ama pozitif kalır. Karar: echo route'u için T0 provisional giriş güvenli (bu subset'te tail yapısal olarak yok), veya T+15 ile daha muhafazakâr.
3. **Hold:** echo route'unda 4h→6h uzatmak avg+tail iyileştiriyor.
4. **İlk adım:** `echo_30_90 + regime` shadow bucket olarak forward izle (T0 ve T+15 iki varyant); yeterli forward N + holdout sonrası live promotion.
5. echo_30_120 ve prebuild varyantları shadow-only kalır (T+15'te ölüyor / tail var).

### Guardrail
Research-only. Live executor / `.env` / order logic / leverage / sizing dokunulmadı.

---

## 29. Full Signal Boost Gauntlet (2026-07-01)

Rapor:
- `reports/research/s34/S34_FULL_SIGNAL_BOOST.md`
- `reports/research/s34/S34_FULL_SIGNAL_BOOST.json`
- Script: `tools/research_s34_full_signal_boost.py`

Kapsam:
- Hour17 confidence / tail / entry / exit tests.
- SHORT_NOISY BTC-confirmed route sweep.
- BUY-side fade portfolio leg test.
- Cross-asset, funding, book, OFI separators.

Guardrail:
- Research-only.
- Live executor / shadow runner behavior / `.env` / leverage / sizing degismedi.

### Hour17 ana sonuc

| Route | No-overlap N | /ay | WR | avg | total | tail100 | mc_p | WF |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hour17 200K current | 63 | 14.0 | 61.9% | +33.6 | +2117.7 | 11 | 0.021 | 5/5 |
| hour17 150K | 73 | 16.2 | 60.3% | +42.3 | +3084.9 | 10 | 0.002 | 5/5 |

150K threshold, current 200K route'a gore daha yuksek total/avg ve daha guclu mc verdi. Live degisikligi yapilmadi; bu bir paper/shadow adayidir.

### Hour17 kalite filtreleri

200K icin en guclu confidence pocket:
- `funding_rate=lo & sync_ratio=hi`: no-overlap N=17, /ay=3.8, WR=94.1%, avg=+119.8, tail100=1, mc_p=0.001, WF 5/5.
- `funding_rate=lo & sync_sell_pre=hi`: no-overlap N=17, WR=82.4%, avg=+116.1, mc_p=0.002.

150K icin en guclu confidence pocket:
- `btc7d=mid`: no-overlap N=29, /ay=6.4, WR=86.2%, avg=+83.0, tail100=0, mc_p=0.0.
- `btc4h=lo`: no-overlap N=28, /ay=6.2, WR=78.6%, avg=+89.0, mc_p=0.0.

Saat ayrimi:
- 17-19 UTC en iyi: 200K N=61, WR=67.2%, avg=+56.3.
- 20-21 ve 22-23 pozitif ama zayif: avg yaklasik +24, mc anlamsiz/sinirda.

Tail veto:
- `exclude_near_funding_30m` anlamli: kept no-overlap avg +44.1, dropped avg -5.5. Funding'e son 30dk kala hour17 zayif.
- `exclude_sat_sun`, `exclude_btc5m_lt_minus50`, `exclude_sync_100_200k` ters calisti veya faydali degil; otomatik veto yapma.
- Dar stoplar hour17'yi bozuyor: stop150 no-overlap avg +4.4, mc 0.405. 300bps stop bile base'den zayif.

Entry/exit:
- T0 veya T+1m en iyi. 200K no-overlap: T0 avg +33.6, T+1m avg +40.7.
- T+30/T+60 gecikme edge'i azaltir.
- 6h hold, 4h/8h/10h'den daha iyi dengede. Profit-lock 200/100 hafif iyi olabilir ama base'e yakin; forward observer ile izlenmeli.

### SHORT_NOISY

En iyi no-overlap route:
- `btc1000k_d5_h180`: N=14, /ay=3.1, WR=92.9%, avg=+110.6, total +1549.0, tail100=1, mc_p=0.003, WF 5/5.
- `btc1000k_d5_h120`: N=14, WR=71.4%, avg=+109.2, tail100=0, mc_p=0.003.

Karar: SHORT_NOISY gercek diversifier. Live degil; once shadow/paper bucket olarak izlenmeli.

### BUY-side fade

| Variant | No-overlap N | /ay | WR | avg | total | mc_p | WF |
|---|---:|---:|---:|---:|---:|---:|---|
| all T0 h45 sl75 | 326 | 72.2 | 45.7% | -5.9 | -1922.5 | 0.954 | 0/5 |
| silence30 T0 label | 177 | 39.2 | 66.7% | +24.6 | +4349.9 | 0.0 | 5/5 |
| silence30 confirm T+30 tradeable | 177 | 39.2 | 40.1% | -10.1 | -1784.7 | 0.994 | 0/5 |
| silence30 + ask_depth_hi T0 label | 41 | 9.1 | 75.6% | +28.0 | +1148.2 | 0.0 | 5/5 |

Kritik yorum: BUY fade, T0 silence label ile guzel gorunuyor ama silence T+30'da bilinir. Tradeable T+30 confirmation oluyor. Bu nedenle BUY fade live/paper promotion adayi degil; ancak T0-knowable ask-depth predictor ayrica arastirilabilir.

### Portfolio

Research portfolio (BUY fade T0 label dahil, lookahead uyarili):
- H17 only: N=63, /ay=14.0, avg +33.6, total +2117.7.
- H17 + SHORT_NOISY: N=74, /ay=16.4, avg +45.8, total +3388.7.
- All three: N=217, /ay=48.1, avg +32.2, total +6983.5.

Canliya yakin yorum: H17 + SHORT_NOISY en temiz pratik genisleme. BUY fade portfoy sonucu lookahead label icerir, live-readiness yok.

### Sonraki karar

1. Hour17 150K veya `funding_rate=lo & sync_ratio=hi` confidence pocket shadow/paper bucket olarak eklenebilir; live degisikligi icin operator sign-off gerekir.
2. SHORT_NOISY `BTC>=1M delay>=5m hold=180m` shadow/paper bucket oncelikli.
3. BUY fade icin yeni T0-knowable predictor arastir: ask-depth, book imbalance, BUY-side sync, pre-OFI. T+30 silence confirmation kullanma.
4. Sizing/tail-budget riski hala acil ve alpha iyilestirmeden bagimsiz.

---

## 28. hour17 Route — LIVE + Shadow + Dashboard Implementasyonu (2026-07-01)

§27'deki deploy adayı (`hour≥17 UTC + regime` hold-6h) shadow + dashboard + **LIVE** olarak eklendi. Operatör onayı alındı (stop + entegrasyon kararları).

### Route tanımı: `LONG_HOUR17_HOLD6H`
- **Gate:** not bull, not EUROPE, regime(btc4h<0 OR btc7d<0), hour≥17 UTC. (Anchor eşiği 200K — mevcut stream.)
- **Giriş:** anchor T0 (slot boşsa hemen, ~T+60s).
- **Hold:** 6h, **erken çıkış YOK**.
- **Stop:** **300bps geniş güvenlik stopu** (operatör onaylı — araştırma edge'i korunur, 40x gap-through kapağı).
- **Entegrasyon:** mevcut T+15 LONG + SHORT route'larıyla **birlikte** (coexist). Tek-pozisyon: hour17 slotu doldurunca T+15 LONG o anchor'da decline eder. BTC≥2M confirm gelirse hour17 LONG → SHORT'a flip (mevcut mantık).

### Değişen dosyalar
| Dosya | Değişiklik |
|---|---|
| `tools/s34_realtime_shadow_runner.py` | `LONG_HOUR17_HOLD6H` observation route (T0 aç, +6h time-exit, erken çıkış/stop yok). Sabitler: `HORIZON_LONG_H6_MS`, `HOUR17_MIN_HOUR=17`. |
| `tools/s34_live_chart.py` | Candidate bucket `C_hour17_hold6h` + **`★ ACTIVE ALPHA` spotlight kartı** (payload `active_alpha`: isim, tanım, canlı+shadow N/WR/TOTAL/avg, açık pozisyon özellikleri). open_positions'a hour/btc4h/7d/entry/notional/exit eklendi. Candidate tracker label düzeltildi (hour17=LIVE). |
| `tools/s34_state_machine_live_executor.py` | **LIVE route.** `hour17_eligible` row'a eklendi; `handle_new_anchor` hour17'yi de kabul eder; `manage_pending`'e T0-açılış bloğu; `open_market_position`'a geri-uyumlu `stop_bps_override` param (mevcut çağrılar etkilenmez). Sabitler: `LONG_HORIZON_H6_MS`, `HOUR17_MIN_HOUR`, `HOUR17_STOP_BPS=300`. |

**DOKUNULMADI:** sizing, leverage, ORDER_NOTIONAL, `.env`, margin, mevcut T+15/SHORT order mantığı. Stop mekanizması aynı (sadece hour17 için bps override).

### Doğrulama (sandbox)
- `py_compile` üç dosya ✓; hour17 identifier wiring ✓.
- Live executor `--once` dry-run: tüm mantık çalıştı, sadece **borsa auth'ta** durdu (`-2015 Invalid API-key/IP` = sandbox IP whitelist dışı, KOD DEĞİL).
- Shadow runner `--once`: EXIT=0, çökme yok.

### ⚠️ OPERATÖR AKSİYONU — canlıya almak için gerekli
Sandbox içinden restart network'ü bozar + borsa IP whitelist dışı. **Sandbox dışı normal PowerShell'de:**
```
powershell -NoProfile -ExecutionPolicy Bypass -File start_eclipse.ps1
```
Bu, shadow runner + dashboard + live executor'ı yeni kodla yeniden başlatır = **hour17 route CANLI ARMED olur.** (Restart = bilinçli arming adımı.)

Doğrulama sonrası kontrol: `status_eclipse.ps1` → live executor alive, dashboard `/api/data` içinde `active_alpha` + `C_hour17_hold6h` bucket görünür; SHADOW panelinde ★ ACTIVE ALPHA kartı (N/WR/TOTAL PnL); ilk hour≥17 + regime cascade'de `open_long_hour17_hold6h` log'u ve dashboard'da active LONG.

**Codex handoff:** `ALPHA_HANDOFF_CODEX.md` (repo root) — hangi alpha, kanıt zinciri, kesin kurallar, kapalı hipotezler, kod haritası, yeni geliştirme fikirleri. Codex'e direkt paste edilir.

### 28b. Leverage 40x→15x + executor death teşhisi (2026-07-01)

**Leverage 15x'e düşürüldü** (`.env` `S34_LIVE_MAX_LEVERAGE=40→15`, operatör kararı). Sebep: 40x'te likidasyon ~%2.3'te → hour17 300bps (%3) stop ve strateji "dip'i tut" mantığı çalışmıyordu (stop likidasyon ötesindeydi). 15x'te likidasyon ~%6.5 → stop'lar (300bps hour17, 150bps diğer) likidasyondan ÖNCE tetiklenir; edge korunur.

Sizing ($32.13 bakiye, 15x, %85 margin, 300bps stop):
| | 40x (eski) | **15x (yeni)** |
|---|---|---|
| margin (riske) | $27.31 | $27.31 (%85 aynı) |
| notional | $1,092 | **$410** |
| likidasyon mesafesi | ~%2.3 | **~%6.5** |
| 300bps stop çalışır mı | ❌ (likidasyon önce) | ✅ (%3'te, %6.5 önce) |
| stop'lanan trade kaybı | ~$27 (likidasyon, %85) | ~$12.3 (%38 hesap) |

> Kalan risk: %85 margin'de bir stop hâlâ hesabın ~%38'i. Daha da güvenli için `S34_LIVE_MARGIN_PCT_ETH` düşürülebilir (operatör kararı — şu an dokunulmadı).

**Executor "hep ölüyor" teşhisi:** `logs/s34_state_machine_live_executor.out.log` → `CRITICAL AUTHENTICATION FAILED: binanceusdm {"code":-2015,"msg":"Invalid API-key, IP, or permissions"}` saniyede bir tekrar → açılışta market yüklenemiyor → restart döngüsü. **Kod hatası değil — Binance API key IP-whitelist / Futures-izni.** Operatör aksiyonu (kullanıcı hallediyor). İkincil latent: `set_leverage_if_possible→resolve_symbol` NotImplementedError (ilk live emirde patlayabilir — auth düzelince izlenecek).

### 28c. Early-Signal + Management Gauntlet (2026-07-01)

Rapor: `reports/research/s34/S34_EARLY_MGMT.json/.md`. Script: `tools/research_s34_early_mgmt_gauntlet.py`.
hour17 200K baz (126 event, 4.5 ay). A-F test bankası, research-only.

**🏆 En güçlü iyileştirmeler (mekanizmaya uygun, live-değil-henüz):**
| Bulgu | Etki | Not |
|---|---|---|
| **B2 LIMIT entry -20/-30bps** (anchor altına pasif) | avg +33.6→**+62.7/+70.4**, WR 62→**70%**, mc 0.0 | Fiyat dip yapıp toparlıyor; maker fill + fee tasarrufu. En büyük entry kaldıracı. Fill varsayımı forward test ister. |
| **C5 scale-in @-100bps dip** (2. birim ekle) | avg +39.8→**+57.0**, WR→**70.6%**, TOT +7182, mc 0.0 | Dip=kapitülasyon dibi; ekle. AMA 2x notional=tail (-400) + likidasyon riski, 15x'te sizing dikkat. |
| **D1 conviction-weighted sizing** (sync_ratio+funding+accel skoru) | flat +2118 → **weighted +6207** (~3x) | Yüksek-güven büyük poz. score2 WR63 avg+51, score3 WR100. |
| **D2 sync_ratio = #1 magnitude predictor** | hi +71.3 vs lo +8.4 (delta **+62.9**) | (BTC+SOL eş-zamanlı sell)/ETH cascade. be_ratio_pre +43.7, funding +20.8. btc5m NEGATİF (-51.6). |
| **A3 cascade RATE (hız) filtresi** | rate_hi WR **71.4%** avg +44.6 mc 0.0 | Hızlı cascade daha iyi bounce = yeni T0 kalite gate'i. |
| **C1 wide trailing (arm200/trail100)** | avg +44.0 (fixed +39.8), WR 65.9 | Geniş trailing hafif iyileştirir; dar trailing bozar. |

**❌ Doğrulanan anti-pattern'ler (yapma):**
- A2 first-liq / çok-erken giriş: WR 57 avg +20.5 — cascade tamamlanmadan girme, T0 bekle.
- Breakeven stop (be50 WR 33%), funding-exit (+24), adaptive-8h (WR 46), tight trailing — hepsi hold'u kesip bozuyor ("dip'i tut" ihlali).
- A1 100K: no-overlap mc 0.074 (marjinal) — 100K çok uzak; 150K (mc 0.004) tatlı nokta.
- F3 cross-asset: SOL own mc 0.168, BTC own mc 0.12 — **edge ETH'e özel**, transfer olmuyor.
- F1 short_noisy hour17 universe içinde N=4 negatif (full short_noisy ayrı evren diversifier'dı; hour17 içinde değil).

**E) Regime:** mild btc7d(>-300) WR 69% worst -184 > deep(<-300) WR 60% worst -448 (derin=fat tail). Her ay pozitif; Nisan zayıf (+2.2), Haziran güçlü (+63.4).

**Sıradaki live-öncesi:** B2 limit-entry + D sync_ratio-conviction-sizing en yüksek EV; ikisi de chronological holdout + forward şart. Scale-in sizing'i 15x'te tehlikeli — dikkatli modelle.

### 28d. Signal Mining — cascade öncesi/anı/sonrası (2026-07-01)

Rapor: `reports/research/s34/S34_SIGNAL_MINING.json/.md`. Script: `tools/research_s34_signal_mining.py`.
30+ microstructure sinyal (bid/ask/book, OI[stale], basis, flow, liq-şekli, vol, funding, price-action) hour17 200K bazda **holdout 70/30** ile tarandı (TRAIN yön seçer, TEST raporlar). PRE/AT=T0, POST=T+5 delayed.

**🏆 Robust YENİ sinyaller (TEST N≥15, delta+, full mc 0.0):**
| Sinyal | Window | Fav | TEST N | TEST WR | TEST avg | delta |
|---|---|---|--:|--:|--:|--:|
| **sync_ratio** (BTC+SOL sell/rn) | pre | hi | 20 | 85.0% | +102.6 | +71 |
| **rv5m** (realized vol) | pre | hi | 38 | 73.7% | +69.0 | +69 |
| **bid depth rebuild 0→5m** | post | hi | 18 | 88.9% | +95.7 | +62 |
| **density24** (24h cascade sayısı) | pre | hi | 33 | 75.8% | +75.4 | +49 |
| **OFI pre-5m** (buyers) | pre | hi | 22 | 81.8% | +87.4 | +44 |
| **be_ratio** (BTC_conc/rn) | pre | hi | 16 | 93.8% | +122.8 | +93 |
| **book imbalance/ba_ratio** (ask-heavy) | pre | lo | 26 | 76.9% | +77.2 | +26 |
| price reclaim/BTC post-5m (lo) | post | lo | 17 | 76.5% | +82.5 | +35 |

> **sync_ratio + rv5m tekrar #1-2** (3. kez doğrulandı). **Yeni:** bid-depth-rebuild (POST, tradeable confirmation, WR 88.9%), density24, OFI-pre. be_ratio hi güçlü AMA non-monotonic — extreme (≥2) hâlâ tail veto (mega). Book **ask-heavy** = bounce iyi (mega F1 ile tutarlı).

**❌ Çürütülen / robust-değil (holdout'ta düştü):**
- **liq/sec rate: delta -86** — §28c A3'teki rate_hi in-sample'dı, **OOS tutmadı**. Cascade hızı güvenilir değil.
- cascade drop magnitude (-75), rn size (-15), liq_count (-25), max_single — şekil/boyut robust discriminator değil.
- funding / spread @T0: hour17 evreninde düşük varyans, split degenerate (TEST N=0).

**Küçük-N güçlü ipuçları (N=6-8, doğrulama gerek):** agg-trade count pre-5m hi, ETH ret pre-1h lo (derin düşüş→iyi bounce), BTC ret 5m lo (BTC de düşüyor→iyi).

**Sentez:** En iyi kompozit T0 conviction = **sync_ratio↑ + rv5m↑ + density24↑ + OFI-pre↑** (+ be_ratio moderate, extreme veto) + POST teyit **bid-rebuild↑**. Bu, §28c'deki sync_ratio-conviction-sizing'i güçlendiriyor. Live-öncesi: bu kompoziti tek skor yapıp holdout+forward.

### 28e. Conviction Composite + Shadow Paper Route (2026-07-01)

Rapor: `reports/research/s34/S34_CONVICTION_COMPOSITE.json/.md`. Script: `tools/research_s34_conviction_composite.py`.
6 robust sinyal TEK skora birleştirildi (sync_ratio + rv5m + density24 + ofi_pre + be_ratio-mod + ask-heavy, 0-6). Eşikler TRAIN medyanı.

**Skor MONOTON + OOS doğrulandı:**
| Skor | N | WR | avg | mc_p |
|--:|--:|--:|--:|--:|
| 0-1 | 25 | ~40% | negatif | — |
| 2 | 31 | 68% | +26 | 0.09 |
| 3 | 32 | 62% | +51 | 0.012 |
| **4** | 23 | **83%** | +98 | 0.0 |
| 5 | 15 | 80% | +72 | 0.004 |

- `score≥3` **TEST-OOS: WR 82%, avg +101, mc 0.0**; `score≥4` **TEST-OOS: WR 90%, avg +113, worst -44.8**.
- **Conviction-weighted sizing (unit=score): flat +2118 → weighted +8936 (~4.2x)** — en büyük kaldıraç.
- `score≥3 + LIMIT -20bps`: WR 79.4%, avg +83.4 (+13 vs market).

**Composite = live hour17'den farkı:** aynı universe; EK conviction skoru + skora-göre-sizing + limit giriş. Live hâlâ ham sabit-boyut market hour17.

**Shadow paper route eklendi (LIVE DEĞİL):**
- `tools/s34_realtime_shadow_runner.py`: `compute_composite_score()` (sabit eşikler) + `LONG_HOUR17_COMPOSITE` route (hour17 + score≥3, T0 giriş, 6h hold, no early-exit). Score + components pozisyonda saklanır.
- `tools/s34_live_chart.py`: bucket `C_hour17_composite` (score≥3) + `C_hour17_composite_s4` (score≥4 high-WR sleeve).
- Doğrulama: py_compile OK; `--once` EXIT=0; payload 18 bucket, composite'ler mevcut.
- Live executor / .env / sizing DOKUNULMADI. Forward veri biriktikçe skor canlıda doğrulanacak.

### 28f. Deep Questions Q1-Q8 (2026-07-01)

Rapor: `reports/research/s34/S34_DEEP_QUESTIONS.json/.md`. Script: `tools/research_s34_deep_questions.py`. hour17 200K, 126 event.

**✅ DOĞRULANAN / actionable:**
- **Q1 Ablasyon:** 6 sinyal de katkı sağlıyor; solo lift sync_ratio(+62.9)>rv5m(+56.0)>be_ratio(+43.1)>ask_heavy(+41.8)>density24(+31.3)>ofi_pre(+28.9). sync_ratio+rv5m çekirdek. Skor additive.
- **Q6 Liq-shelf = YENİ 7. sinyal:** fiyat altında (0-2%) kümelenmiş 24h likidasyon çok → shelf_hi WR 71.4% avg +59.1 vs lo +20.6, **delta +38.5, mc 0.0.** Conviction skoruna eklenebilir.
- **Q8 Funding veto (YENİ):** cascade funding'e <60m ise **ÖLÜ** (WR 45%, avg -7.0, mc 0.602). 60-240m +38.3, >240m +56.3. → funding'e <60m = VETO.
- **Q4 SHORT'un KENDİ conviction'ı:** btc_size (BTC confirm büyüklüğü) **delta +154.9** (dominant), be_ratio +56.4, sync_ratio +50.2. rv5m/imb/ofi SHORT için alakasız. SHORT skoru ayrı kurulmalı.
- **Q3 Conviction-weighted sizing 15x'te GÜVENLİ:** weighted 10% unit → final 1.47x, **max DD sadece -4.8%** (flat 1.27x/-2.6%). No-overlap gerçekçi; sizing lever yeşil ışık.
- **Q7 Edge STABİL/güçleniyor:** fold'lar WR 70.6/70.6/83.3; ilk-yarı +61.2 vs son-yarı +86.8 → **decay YOK, son dönem daha güçlü.**

**❌ ÇÜRÜTÜLEN (kritik düzeltmeler):**
- **Q2 Limit-entry -20bps MARKET'ten KÖTÜ:** fill sadece %36 → kaçan trade'lerle **EV/signal +34.5 vs market +74.2.** §28c/§28e'deki limit +62.7 survivorship'ti (sadece dolanları saymış). → **MARKET entry kullan, limit değil.**
- **Q3 Uzun hold (8h) BOZAR:** conditional 8h WR 65.2/avg +70.4 < fixed 6h WR 73.9/+74.2. Sabit 6h kalır.
- **Q5 Skor hour17'yi DEĞİŞTİRMEZ:** all-hours score>=4 no-overlap mc 0.106; hour<17 subset avg +2.2 mc 0.458 (ölü), hour>=17 WR 83.3% mc 0.0. **hour17 şart, conviction onun içinde kalite katıyor.**

**Rafine tarif:** hour17 + conviction skoru (7 sinyal: +liq-shelf) + funding-veto(<60m) + conviction-weighted-sizing (15x güvenli) + **MARKET giriş** + sabit 6h. SHORT diversifier'a ayrı btc_size-öncelikli conviction.

**Shadow route güncellendi (2026-07-02):** `compute_composite_score` artık 0-7 (liq-shelf 7. sinyal, eşik $2.775M = hour17 medyanı) + `funding_veto` flag (<60m → skip). `LONG_HOUR17_COMPOSITE` route funding-veto uygular. Dashboard label'lar 7-sinyal'e güncellendi. Doğrulama: py_compile OK, `--once` EXIT=0, örnek skorlar 4-6/7, veto min2fund=53.6'da True. Live/.env/sizing DOKUNULMADI.

### 28g. SHORT Conviction Gauntlet (2026-07-02)

Rapor: `reports/research/s34/S34_SHORT_CONVICTION.json/.md`. Script: `tools/research_s34_short_conviction.py`.
SHORT_NOISY 36 event (BTC≥500K). **DİKKAT: küçük N — hepsi directional, forward şart.**

**✅ Bulgular:**
- **btc_size conviction monoton:** BTC≥500K WR 69% avg+81; ≥1M WR 75% avg+91; ≥2M **WR 100% avg+166** (N=9). Hold 120-180m. (btc_size Q4'te delta+155.)
- **SHORT hour = LONG'un TERSİ:** h13-17 (US erken öğleden sonra) **WR 100% avg+182 mc 0.0** (N=11); h17-24 **ÖLÜ** (WR 20% avg-99). LONG hour≥17 ister, SHORT h13-17 ister — güzel asimetri.
- **SHORT funding = LONG'un TERSİ:** <60m WR 100% avg+247 (N=4); >240m ölü. (LONG'da <60m veto'ydu.)
- shelf-above hi (üstte likidasyon=direnç) WR 80% > lo 70% (küçük N).
- SHORT conviction skoru (btc_size+be_ratio+sync_ratio) score≥3 WR 100% avg+180 (N=6) — ama TEST N=2, OOS doğrulanamaz.

**🔴 KRİTİK realizm uyarısı (S5):**
- entry @noisy_ts: avg **+91** (mc 0.012) — AMA bu LOOKAHEAD (BTC confirm daha sonra biliniyor).
- entry @BTC-confirm (tradeable): avg **+32.6, mc 0.176 (anlamsız)**. → **SHORT edge tradeable girişte çok zayıflıyor.** Görünenden kırılgan.

**Portföy (S7):** LONG-composite only +5002; SHORT ekleyince non-overlap +2589 (short nadir, 7 event). SHORT modest diversifier.

**Karar:** SHORT conviction gerçek ama küçük-N + tradeable-entry kırılgan. Doğru hamle = confirm-entry + hour13-17 + BTC≥1M/2M + 180m hold ile **forward paper route** (in-sample yetersiz). Live değil.

### 28h. Frontier Gauntlet — prediction/navigation/in-data/signal/meta (2026-07-02)

Rapor: `reports/research/s34/S34_FRONTIER.json/.md`. Script: `tools/research_s34_frontier_gauntlet.py`. hour17 200K composite, 126 event.

**P PREDICTION — 100K erken giriş (frekans genişletme):**
- 100K mini (hour17): 35.5/ay WR 61.2% mc 0.002. **score≥3: 21.5/ay WR 64.9% avg +55, mc 0.0** (~1.5x 200K frekansı).
- **Sürpriz: fizzle eden (200K'ya BÜYÜMEYEN) mini'ler daha iyi bounce** (WR 67.9% vs büyüyenler 53.9%). Küçük izole liq = overreaction = daha iyi reversion.

**N NAVIGATION — state grid:**
- **Derin btc7d rejiminde score ŞART:** h17-19 deep7d s<4 WR **30%** (ölü) vs s≥4 WR **80% avg+120**; h20-23 deep7d s≥4 WR **84.6% mc 0.0**. Composite en çok derin-düşüş rejiminde değerli.
- **N2 momentum:** kazanan cascade sonrası sonraki de iyi (WR 67.9% mc 0.0); kaybeden sonrası zayıf (WR 59% mc 0.314). Pozitif otokorelasyon → kazançtan sonra devam, kayıptan sonra temkinli.

**D IN-DATA:**
- **Cross-asset simultane güçlü:** ETH+BTC(≥500K) WR 75% +75; **ETH+SOL(≥100K) WR 78.9% +96 mc 0.012**; ETH-only WR 62%. Senkron cascade = kapitülasyon = iyi bounce.
- Storm(5+/gün) WR 67% +64 ve izole(1-2) WR 69% iyi; orta (busy 3-4) en zayıf.
- **Multi-horizon: 6h KESİN optimal** — WR 2h→6h yükselir (62→74.7%), 8h'te düşer (59.5%). 6h hold doğrulandı.

**S SIGNAL SEARCH — yeni:**
- **whale_LO (küçük ort. trade = retail) daha iyi:** WR 73% +64 mc 0.0 vs whale_HI WR 57% +15 mc 0.184. Retail-driven cascade = overreaction = iyi bounce; whale = informed = devam.
- time-since-last <2h: WR 69% +56 mc 0.0 (clustering iyi).

**M META — threshold sweep (para tablosu):**
| score≥ | /ay full | /ay noov | WR | avg | mc_p noov |
|--:|--:|--:|--:|--:|--:|
| 2 | 20.9 | 10.0 | 71.3% | +57 | 0.0 |
| 3 | 15.8 | 7.8 | 74.6% | +70 | 0.002 |
| 4 | 11.1 | 5.1 | 76.0% | +78 | 0.032 |
| 5 | 5.8 | 3.1 | 84.6% | +104 | 0.01 |
- Monoton. **score≥2 = frekans opsiyonu (20/ay), score≥5 = premium sleeve (WR 85%).** deep7d×s≥4 en iyi hücre (WR 82%).

**Yeni actionable:** (1) 100K composite frekans genişletme adayı, (2) whale_lo yeni sinyal, (3) navigation: deep-regime'de yüksek-skor zorunlu + momentum-after-win, (4) cross-asset simultane filtre, (5) 6h kesin optimal. Hepsi research-only, live/paper'a dokunulmadı.

### 28i. Horizon Gauntlet + Composite v3 (2026-07-02)

Rapor: `reports/research/s34/S34_HORIZON.json/.md`. Script: `tools/research_s34_horizon_gauntlet.py`.

**✅ HOLDOUT-DOĞRULANAN (v3'e alındı):**
- **whale_lo HOLDOUT geçti:** TEST WR **94.1%** avg +116.5 mc 0.0 (N=17 OOS) vs whale_hi WR 57% mc 0.124. → composite'e **8. sinyal** olarak eklendi (avg agg-trade <$6440 = retail).
- **100K composite HOLDOUT geçti:** TEST WR 72.4% mc 0.0; no-overlap **11.8/ay** WR 66% avg +54.6 mc 0.0. Frekans genişletme adayı (200K'nın ~1.5x).
- **Edge ROBUST:** block-bootstrap (otokorelasyon-aware) score≥3 noov → 5% CI **+26.3 bps**, **P(avg<0)=0.000.** Fluke değil.

**Feature interaction (I):** her iki-sinyal-birden = süper. sync+whale_lo WR **86.7%**, shelf+be WR 86.4%, sync+shelf WR 82.8%. Additive yapı doğrulandı.

**Meta-veto (Z):** score≥3 içinde KAYBEDEN-pocket yok — her feature'ın "removed" tarafı bile pozitif (+41..+102). Ek veto YOK; skor tek discriminator (funding-veto kalır).

**Swing horizon (H):** 6h WR 75.9% (en iyi WR); 12h avg +84 ama WR 66.7%; 24h +66 mc 0.03; **48h -31 mc 0.706 (ÖLÜ).** 6h optimal kesin; 24h ötesi yok.

**Time-machine (T):** SHORT 13-17 + confirm-entry + BTC≥1M → WR **81.8% mc 0.006** (N=11, ungated'den iyi). Portföy (LONG17-23+SHORT13-17) noov 8.7/ay WR 69%. SHORT kalite diversifier ama nadir.

**v3 uygulandı:** `compute_composite_score` 0-8 (whale_lo 8. sinyal, eşik $6440). Dashboard label 8-sinyal. py_compile OK, `--once` EXIT=0. Live/.env/sizing DOKUNULMADI.

**Sıradaki adaylar:** 100K frekans route (paper), SHORT 13-17 confirm route (paper), sync+whale_lo premium sleeve.

### 28j. Paper Routes eklendi + shadow bug fix (2026-07-02)

- **`LONG_HOUR17_100K_COMPOSITE`** shadow route: 100-200K mini cascade + hour17 + regime + composite score≥3 (funding-veto). Entry T0, 6h hold. `run_once`'a 100K reconstruction loop + `open_100k_composite_for_anchor` handler. Dashboard `C_hour17_100k_composite`.
- **`C_short_noisy_1317`** dashboard bucket: mevcut `SHORT_NOISY_BTC1M_D5_H180` route'unun entry-hour 13-17 UTC alt-kümesi (yeni route kodu yok, sadece filtre). OOS WR 82.
- **BUG FIX (shadow, önemli):** `mark_at` `SELECT price` → `SELECT mark_price` (kolon adı yanlıştı). Bu bug SHORT_NOISY/BUY_FADE girişlerini çökertiyordu (paper tracking eksik kalıyordu). Artık düzgün. Sadece shadow — live risk yok.
- Doğrulama: py_compile OK, `--once` EXIT=0 (BUY_FADE girişi başarıyla açıldı), payload 20 bucket, iki yeni bucket mevcut. Live/.env/sizing DOKUNULMADI.

### 28k. Puzzle Gauntlet — mekanik + bind-roads + kapsama (2026-07-02)

Rapor: `reports/research/s34/S34_PUZZLE.json/.md`. Script: `tools/research_s34_puzzle_gauntlet.py`.

**🔬 MEKANİK (NEDEN çalışıyor — çözüldü):**
- **A1 Bounce = YAVAŞ mean-reversion drift**, ani snap DEĞİL: avg MFE **+151**, avg MAE -84, median time-to-peak **~199dk (3.3h)**, sadece %32 peak<=2h. MFE(+151) >> yakaladığımız(+64) → giveback var.
- **A3 whale mekaniği:** retail(whale_lo) drop -33/bounce +72 vs whale drop -41/bounce +50. **Retail overreaction reverts; informed selling persists.** Mekanizma net.
- **A2 hour çekirdeği:** h19-20 UTC (US öğleden sonrası) WR **79%** en iyi. Edge = US-afternoon retail-overreaction reversion.

**🆕 YENİ SİNYAL:** B1 cascade **"gentleness"**: grind (küçük drop) WR **79%** > sharp drop WR 69%. Nazik cascade daha iyi bounce (A3 ile tutarlı).

**🛣️ YOL:** C2 profit-target **+200bps + 6h** fixed-6h'i hafif geçiyor (TOT +5770 vs +5438) — MFE spike'larını yakalıyor. target100/150 çok dar.

**🔗 YOLLARI BAĞLA (kritik bulgu):**
- **D2 route'lar KORELE:** 56 trade-günün %61'inde ≥2 route aynı gün fire → çeşitlendirme SINIRLI (aynı storm'larda kümeleniyor).
- **D1 naive FIFO scheduler ALT-PERFORMANS:** birleşik 13.3/ay TOT +3634 < long_comp tek başına +5438. 100K (sık, düşük-kalite) slotu kapıp yüksek-conviction comp'u pre-empt ediyor. → **Conviction-öncelikli scheduler şart (master navigator):** boş slotta en yüksek-skor route kazanmalı. Bind-roads'un asıl aksiyonu bu.

**🧩 PUZZLE KAPANDI:**
- **E1 00-13 UTC LONG boşluğu GERÇEKTEN BOŞ** (h13-17 mc 0.306 ölü, h00-07 mc 0.088 marjinal). hour17 gate doğru.
- **E2 BUY-side KESİN ÖLÜ** (conviction ile bile: WR 49.5 avg-17 mc 0.992). Short-squeeze fade kapandı.

**🛡️ ROBUSTLUK:** F1 purged 5-fold **5/5 pozitif** [50,23,91,46,109]. F2 edge hem zayıf-geçiş (btc7d -100..0 mc 0.028) hem güçlü-bear (mc 0.0) rejiminde yaşıyor.

**Actionable:** (1) conviction-priority scheduler (master navigator) = bind-roads sonraki adım, (2) cascade-gentleness sinyal adayı, (3) +200 profit-target, (4) 00-13 & BUY kesin kapalı — buralara bakma. Live/paper'a dokunulmadı.

### 28l. Master Navigator — conviction-priority scheduler (2026-07-02)

Rapor: `reports/research/s34/S34_MASTER_NAVIGATOR.json/.md`. Script: `tools/research_s34_master_navigator.py`.
Route havuzu 193 event (LONG_comp 85, LONG_100k 97, SHORT_1317 11). Tek-slot, LOOKAHEAD YOK.

**🔧 ÖNEMLİ DÜZELTME (Puzzle D1 yaniltıcıydı):** Fair no-overlap bazda **route birleştirmek YARDIM EDİYOR:**
| Politika | /ay | WR | TOTAL | mdd | mix |
|---|--:|--:|--:|--:|---|
| P0 long_comp tek başına | 9.3 | 73.8% | +2462 | -176 | comp42 |
| **P1 FIFO union (naive)** | 13.2 | 70.0% | **+3654** | -264 | comp17/100k39/short4 |
| P2 admit>=4 | 10.4 | 66.0% | +2985 | -266 | — |
| P2 admit>=5 | 7.1 | 71.9% | +1881 | -227 | — |
| **P3 route-priority** (comp≥3,short,100k≥5) | 10.8 | **73.5%** | +2857 | **-222** | comp31/100k14/short4 |

- **Naive FIFO union total'i en yüksek (+3654)** çünkü 100k event'leri +EV (avg+49); onları filtrelemek para KAYBETTİRİYOR. Puzzle D1'deki "100k pre-empt sorunu" abartılıymış (100k +EV).
- P3 route-priority: total daha düşük ama **WR 73.5% + en düşük mdd (-222)** = en iyi risk-ayarlı.

**💰 ASIL KALDIRAÇ = conviction-weighted SIZING (scheduling değil):**
- P2_admit>=4: flat +2985 → **weighted +8730** (per-unit +62.8, ~2.9x)
- P3: flat +2857 → **weighted +7155**

**Karar:** Bind-roads çözümü = **geniş-admission (FIFO ya da route-priority) + conviction-weighted sizing.** Akıllı scheduling'den çok SIZING para getiriyor. Drawdown küçük (-176..-353 bps, deploy-safe). Öneri: risk-ayarlı için P3 (WR 73.5, mdd -222) + weighted sizing (+7155). Live/.env/sizing DOKUNULMADI.

### 28m. SQL Meta-Analiz + Temiz View + Konsensüs Kompoziti (2026-07-02)

Kaynak: `S34_ALL.db`. Scriptler: `analyze_s34_all_sql.py`, `build_s34_clean_view.py`, `research_s34_consensus_composite.py`. Raporlar: `S34_ALL_INSIGHTS.md`, `S34_ALL_CLEAN.md`, `S34_CONSENSUS_COMPOSITE.md`.

**Meta-analiz (S34_ALL_INSIGHTS):**
- **Consensus POZİTİF (çapraz-rapor):** funding(%64 pozitif), hour17(%98), shelf/whale/rv5m(%100/78/71), regime/sync(%77-88 anlamlı).
- **Consensus MEZARLIK:** buy-side(%9 pozitif/25 rapor), reversal(%0), fade(%28), cross-asset(%30 anlamlı) → çelişki-bulucu: buy 5:159, fade 5:134, cross 4:101, short 43:235 ölü. hour17 218:5 kaya gibi.
- **Mezarlık: 613 reddedilen** (BUY_REVERSAL_SHORT tüm param %0). **Overfit dedektörü 4590 flag** — tepesi EVENT_CHAIN/FUNDING_EXTREME (total bps DEĞİL, veri-kalitesi bulgusu).
- **Research-vs-paper:** LONG_SILENCE gerçek %44 << research → lookahead gap CANLIDA doğrulandı; SHORT_NEITHER gerçek %71 en iyi.

**Temiz view (research_clean):** 43,458 satır (795 bozuk-rapor atıldı), trust 0-4 (trust=4: 536 satır). **⚠️ Governance uyarısı: high-trust ≠ tradeable** — trust=4 leaderboard tepesi silence/hold-all (+9546) ama bunlar LOOKAHEAD (trust N+mc_p ölçüyor, lookahead değil). WR-ölçeği tutarsız (0-1 vs 0-100) normalize edildi.

**🏆 Konsensüs kompoziti (composite OVER-PARAMETERIZED çıktı):**
| Variant | sinyal | noov /ay | full WR | TEST WR | not |
|---|--:|--:|--:|--:|---|
| v3 equal8 ge3 (mevcut) | 8 | 9.3 | 74.1% | 78.1% | baz |
| **lean4 ge2** (sync+rv+shelf+whale) | 4 | 9.3 | 72.6% | 78.1% | v3'e EŞDEĞER, yarı sinyal |
| **min3 ge2** (sync+shelf+whale) | 3 | 11.0 | **82.0%** | 86.4% | v3'ü GEÇTİ, 3 sinyal |
| weighted ge5 (konsensüs-ağırlık) | 8w | 9.3 | 74.7% | 80.6% | noov WR 76.2 (v3'ten iyi) |
| min3 ge3 / lean4 ge3 | 3/4 | 3.5/4.0 | 93.8/84.2% | 100/86.4% | premium sleeve |

> **Ekstra 4 sinyal (d24/ofi/be/imb) neredeyse HİÇBİR ŞEY eklemiyor.** sync+shelf+whale(+rv) tüm edge'i taşıyor. **min3 (3 sinyal) v3'ü (8 sinyal) WR'da geçiyor (82 vs 74)** → daha az parametre = daha az overfit = daha robust.

**Karar:** Composite sadeleştirilmeli — **lean4 veya min3** bir sonraki versiyon adayı (aynı/daha iyi, daha robust). v3 (8-sinyal) forward-validation'ı bozmamak için şimdilik paper'da kalıyor; lean/min3 aday olarak kayıtlı. Live/.env/sizing DOKUNULMADI.

### ⚠️ Hatırlatma: forward-validation
§27 predictor tek time-of-day feature (US-afternoon reversion, 4.5 ay). Operatör canlıyı armed etti; yine de shadow bucket paralel izlenmeli, birkaç hafta forward sonrası edge teyit edilmeli. Açık URGENT: per-trade margin oversize (60x tail-budget üstü) hâlâ çözülmedi — hour17 canlıyken bu risk geçerli.

---

## 27. T0 Hold Predictor — CASE ÇÖZÜLDÜ: Yüksek-Freq Deploy Adayı (2026-07-01)

Rapor: `reports/research/s34/S34_SILENCE_PREDICTOR.json` + `.md`
Script: `tools/research_s34_silence_predictor.py`
Amaç: T0'da (cascade anında) bilinen feature ile hangi cascade'in 6h tutunca kazanacağını tahmin → yüksek freq + kontrollü tail. Kronolojik 70/30 holdout (eşik TRAIN, rapor TEST).

### 🔑 Silence T0'da tahmin EDİLEMİYOR — ama getiri edilebiliyor

S4: hiçbir T0 feature silence'ı öngörmüyor (tüm lift ≈0). AMA bazı feature'lar hold-6h getirisini **doğrudan** öngörüyor. En güçlüsü **`hour`** (OOS lift +35.7, TEST mc_p 0.005).

Feature OOS lift sıralaması (TRAIN-best tercile → TEST):
| Feature | best-bin | OOS lift | TEST WR | mc_p |
|---|---|---:|---:|---:|
| **hour** | hi (≥17 UTC) | **+35.7** | 66.7% | 0.005 |
| sync_k | mid | +27.7 | 68.6% | 0.06 |
| btc7d | mid | +23.8 | 69.6% | 0.049 |
| btc4h | lo | +23.7 | 61.9% | 0.129 |
| rn | lo (küçük cascade) | +15.8 | 64.3% | 0.077 |
| n2h | hi | +14.0 | 65.8% | 0.094 |
| be_ratio_pre | hi | −21.9 | — | (kötü=veto) |
| btc_conc_pre | hi | −32.1 | — | (kötü=veto) |

### ✅ ÇÖZÜM: `hour ≥ 17 UTC` predictor

Predictor = cascade US öğleden sonrası/akşamı (17-23 UTC). Filtre eşiği TRAIN'de seçildi (hour≥17), TEST'te doğrulandı.

| Config (hold 6h, erken çıkış YOK) | /ay | WR | avg | TOTAL | mc_p | wf | net/mo@10 | mdd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baz: tüm cascade (150K) | 100 | 59.3% | +20.3 | +9207 | 0.003 | 5/5 | — | — |
| hour=hi FULL (150K) | 34.5 | 61.5% | +40.8 | +6360 | 0.0 | 4/5 | — | — |
| hour=hi TEST-OOS (150K) | 35.4 | 66.7% | +47.1 | +2259 | 0.005 | 4/5 | — | — |
| **hour=hi no-overlap (150K)** | **16.2** | 60.3% | +42.3 | +3085 | **0.003** | **5/5** | **+602** | **-391** |
| hour=hi no-overlap (200K) | 13.9 | 61.9% | +33.6 | +2118 | 0.021 | 5/5 | +399 | -356 |
| hour=hi & sync_k=mid noov | 6.2 | 78.6% | +58.8 | +1647 | 0.008 | 5/5 | — | — |

> **hour=hi filtresi hold-all'ı deploy-edilebilir yapıyor:** hold-all no-overlap mc 0.267 (anlamsız) idi → hour=hi ile mc **0.003**, wf 5/5, +602 net bps/ay @10bps. **16/ay** (narrow echo'nun ~2 katı frekans), OOS'ta hayatta.

### Stop yine ZARARLI — ama gerek yok

hour=hi + stop200 → mc 0.105; stop150 → mc 0.247. Stop edge'i kesiyor (dip-recover). AMA hour=hi no-overlap doğal **mdd sadece -391** — hour filtresi zaten derin-tail cascade'leri eliyor. **Stop gereksiz.**

### Yapısal dersler (kesinleşti)
1. **Erken çıkışı kaldır** (noisy'de çıkma, tut) — §26.
2. **Stop koyma** — edge dip'ten toparlanmayı gerektiriyor; hour-filtreli mdd zaten küçük.
3. **Kalite/frekans için hour-of-day** en güçlü T0 predictor; be_ratio_pre / btc_conc_pre yüksekse VETO.

### 🎯 ÖNERİLEN DEPLOY ADAYI

**`ETH SELL cascade ≥150K, not bull, not EU, regime(btc4h<0 OR btc7d<0), hour≥17 UTC → LONG, hold 6h, erken çıkış YOK, stop YOK`**
- ~16/ay, WR 60%, +42 gross / +32 net@10bps, TOTAL +3085, OOS mc 0.003, wf 5/5, mdd -391.
- Yüksek-frekans + robust + kontrollü drawdown = aranan aday.
- Mevcut live gate ile uyumlu (US13-14 bloğu 17-23'ü etkilemez).

**İki live seçeneği:**
- A) Yüksek-freq: `hour≥17` core (16/ay, WR 60, mdd -391).
- B) Yüksek-WR: `hour≥17 & sync_k=mid` (6/ay, WR 78.6, tail 3) — narrow ama çok temiz.

### Sonraki adım (canlı öncesi)
Predictor tek time-of-day feature (US-afternoon reversion — bu 4.5 ay örneğinde). Forward shadow doğrulama şart (regime dönebilir). Öneri: `hour≥17` core'u shadow bucket olarak aç, 4-8 hafta forward + holdout tekrar → operator sign-off → live.

### Guardrail
Research-only. Live executor / `.env` / order logic / leverage / sizing dokunulmadı.

---

## 26. Silence-Core FINAL Gauntlet + Falsification — KESIN SONUÇ (2026-07-01)

Rapor: `reports/research/s34/S34_SILENCE_CORE_FINAL.json` + `.md`
Script: `tools/research_s34_silence_core_final.py`
Amaç: Ana alfa (silence-LONG) çekirdeğini yüksek frekansta canlıya hazırlamak + ciddi çürütme.
Eşikler: 100/150/200K ayrı reconstruct, hold 6h, FEE 5bps.

### 🔴 KRİTİK: "silence" ana alfası büyük ölçüde LOOKAHEAD idi

"silence" ancak T+30dk sonra bilinir → T0'da silence'a bahis lookahead. Gerçek mekanizmalar test edildi:

| Giriş mekanizması (200K, regime, 6h) | N | /ay | WR | avg | TOTAL | mc_p |
|---|---:|---:|---:|---:|---:|---:|
| ideal-silence (LOOKAHEAD) | 142 | 31.4 | 63.4% | +42.0 | +5960 | 0.001 |
| **hold-all (erken çıkış YOK, tradeable)** | 373 | 82.5 | 60.6% | +20.8 | **+7755** | **0.005** |
| provisional early-exit (mevcut live) | 373 | 82.5 | 30.0% | -0.4 | -137 | 0.514 |
| confirm-T30 (silence sonra gir) | 142 | 31.4 | 50.0% | +0.6 | +91 | 0.445 |
| T+15 bounce | 121 | 26.8 | 49.6% | +10.7 | +1298 | 0.204 |

> **ASIL KATİL = ERKEN ÇIKIŞ.** Follow-on likidasyonda çıkmak = kapitülasyon dibinde satmak (follow-on genelde dip, sonra toparlıyor). Erken çıkışı kaldırıp 6h **tutunca** edge geri geliyor: hold-all +7755 mc 0.005. Provisional (mevcut live mekanizması) +7755'i -137'ye çeviriyor. **silence filtresi ≈ "erken çıkışı tetiklemeyen" cascade'ler** — silence özel değil, erken çıkış zararlı.

### Random kontrol hold-all'ı DOĞRULUYOR

F1: random-entry same-regime hold-6h = -1654 (negatif downtrend beta); cascade hold-6h = +9546. **Cascade timing gerçek edge katıyor** (provisional ≈ random idi, hold-all değil).

### 🟡 AMA hold-all deploy edilebilir DEĞİL — kırılgan

Çürütme testleri hold-all'ı da büyük ölçüde eliyor:

| Test (150K, hold-all) | Sonuç | Yorum |
|---|---|---|
| stop ekle | stop150 mc 0.047, stop100 mc 0.035; TOTAL +9223→+4984 | Stop edge'i kesiyor (aynı dip-recover mekanizması) — **edge tail'i tutmayı ŞART koşuyor** |
| **no-overlap (tek pozisyon)** | 36.9/ay, TOTAL +1156, **mc 0.267** (anlamsız), pnl/mo@10=+71 | Gerçekçi tek-pozisyonda anlamlılık ölüyor |
| cost fee10 | pnl/mo@10 ≈ breakeven | +20 gross ince; maliyet yiyor |
| time-split | ilk +2402 (mc 0.158), ikinci +2582 (mc 0.095) | İkisi de tek başına anlamsız |
| top-3 removed | +3159 ama **mdd -22963** | Birkaç dev kazanan taşıyor; tek felaket path |
| no-regime | +2698 mc 0.221 | Regime yardım ediyor ama yetmiyor |
| tail | tail_n 66-134, mdd -3132/-4116 | 40x kaldıraçta deploy-güvenli değil |

> hold-all raw (overlapping) anlamlı (mc 0.005) ama: stop edge'i kesiyor, no-overlap anlamsız yapıyor (mc 0.27), maliyet sonrası breakeven, birkaç dev kazanana bağlı. **Yüksek-frekans versiyonu robust/deploy-safe DEĞİL.**

### ✅ PERFECT RESULT (dürüst): Deploy-hazır alfa NARROW olan

Sıkı testten geçen tek gerçek edge, önceki §24'teki **`echo_30_90 + regime`**:
- Gerçekçi T+15 girişte bile hayatta (mc 0.004, WR 69%), tail 0, PAPER_CANDIDATE.
- **Düşük frekans (~7/ay) tesadüf değil — robust, tail-free edge GERÇEKTEN nadir.**
- §25'teki "over-filter yapıyoruz" eleştirisi kendisi lookahead artefaktıydı: narrow route'lar over-filter değil, reversion'ın gerçek+tradeable olduğu nadir alt-kümeyi izole ediyor.

### İki değerli canlı çıkarım

1. **Mevcut live "noisy'de erken çıkış" kuralı edge SIZDIRIYOR olabilir.** hold(+7755) vs early-exit(-137). Live LONG route'unda erken çıkışı kaldırıp noise'a rağmen tutmayı test et (operator sign-off). Bu tek başına en yüksek-etki değişiklik.
2. **Daha çok frekans için TEK dürüst yol: T0'da bilinebilir bir "silence/kalite predictor".** Hangi cascade'in tutmaya değeceğini cascade anında tahmin eden feature bul (prebuildup, book depth, OFI, be_ratio, vd, rn). Eşik düşürmek İŞE YARAMIYOR (tail-ağır gürültü ekliyor). Gerçek araştırma cephesi bu.

### Karar
- Yüksek-frekans silence/hold-all core **live'a ALINMAZ** (kırılgan, no-overlap'ta anlamsız, tail deploy-unsafe).
- Deploy-hazır alfa = `echo_30_90 + regime` (narrow, §24). Shadow'da forward izle → operator sign-off → live.
- Öncelikli live-relevant test: early-exit kuralını kaldırmanın etkisi (ayrı gauntlet).
- Arka planda geliştirme: T0-knowable silence predictor araştırması.

### Guardrail
Research-only. Live executor / `.env` / order logic / leverage / sizing dokunulmadı.

---

## 25. Alpha Attribution — "Ana Alfa Ne?" (2026-07-01)

Rapor: `reports/research/s34/S34_ALPHA_ATTRIBUTION.json` + `.md`
Script: `tools/research_s34_alpha_attribution.py`
Metrik: **TOTAL** = dönem toplam net bps (asıl para); pnl/mo = aylık ortalama net bps. WR değil.
Evren: 597 ETH SELL 200K anchor, 4.52 ay.

### KESIN SONUÇ: Ana alfa = SILENCE LONG

RAW cascade alfa DEĞİL: her cascade LONG 4h → 132/ay, WR 54.8%, TOTAL +1067, **mc_p 0.389 (anlamsız)**.

Tek-filtre toplam-PnL (hangisi para getiriyor):
| Filtre (solo) | N | /ay | WR | avg | TOTAL | mc_p |
|---|---:|---:|---:|---:|---:|---:|
| **silence** | 255 | 56.4 | 60.4% | +28.3 | **+7225** | 0.0 |
| not_US1314 | 498 | 110 | 56.0% | +7.9 | +3910 | 0.098 |
| echo3090 | 209 | 46.2 | 59.3% | +17.9 | +3751 | 0.051 |
| prebuild | 395 | 87.3 | 57.0% | +6.0 | +2372 | 0.22 |
| regime | 482 | 107 | 57.1% | +3.6 | +1716 | 0.301 |
| score3 | 226 | 50 | 57.5% | +5.7 | +1296 | 0.296 |
| sync200 | 349 | 77 | 54.4% | +2.5 | +872 | 0.352 |

> **silence tek başına tüm parayı taşıyor** (TOTAL +7225, tek anlamlı solo mc_p 0.0). Diğerleri solo anlamsız.

**Leave-one-out (çekirdek = not_bull+not_EU+silence+regime, TOTAL +5037):**
- `− silence` → mc_p **0.141** (edge ÖLÜR)
- `− regime` → mc_p 0.005, TOTAL +4638 (edge YAŞAR)
- `− not_bull` / `− not_EU` → TOTAL artar (bu iki filtre marjinalde para KAYBETTİRİYOR bu evrende)

> **silence load-bearing; regime sadece kalite booster.** Onaylandı.

### Over-filter yapıyoruz — WR yükselirken TOTAL para düşüyor

Kümülatif stack (her filtre eklendikçe):
| Stack | /ay | WR | avg | TOTAL |
|---|---:|---:|---:|---:|
| +silence | 40.0 | 59.7% | +25.6 | +4638 |
| +regime | 31.4 | 63.4% | +35.5 | +5037 |
| +sync200 | 18.8 | 56.5% | +28.0 | +2377 |
| +not_US1314 | 15.9 | 58.3% | +34.7 | +2499 |
| +not_MonWed | 10.6 | 70.8% | +62.7 | +3010 |
| +score3 | **1.5** | 85.7% | +101 | **+709** |

> Filtre ekledikçe WR 60→86'ya çıkıyor ama TOTAL para +4638 → +709'a düşüyor (frekans öldüğü için). **Narrow route'lar (echo 7/ay, full_live 1.5/ay) WR şampiyonu ama para getirmiyor.** Sync200 marjinalde para kaybettiriyor (+4638→ eklenince stack bozuluyor).

### Eşik düşürmek frekans+parayı birlikte artırıyor

| Eşik + silence + regime | /ay | WR | avg | TOTAL | mc_p | wf |
|---|---:|---:|---:|---:|---:|---:|
| **100K** | **61.0** | 62.7% | +31.5 | **+8693** | 0.0 | 5/5 |
| 150K | 40.9 | 61.6% | +24.4 | +4521 | 0.004 | 4/5 |
| 200K | 31.4 | 63.4% | +35.5 | +5037 | 0.001 | 5/5 |
| 300K | 22.8 | 66.0% | +46.0 | +4738 | 0.001 | 5/5 |

> **100K + silence + regime = 61/ay, WR 62.7%, TOTAL +8693** — en yüksek hem frekans hem toplam para. (Not: tail_n=32 — yönetilmeli; no-overlap gerçekliği ayrı.)

### Portföy (non-overlapping, gerçekçi)
| Portföy | /ay | WR | avg | TOTAL | pnl/mo | mc_p |
|---|---:|---:|---:|---:|---:|---:|
| core_long (sil+reg+US+MW, 200K) | 17.0 | 72.7% | +55.2 | +4247 | 939 | 0.0 |
| short_noisy (BTC1M) | 5.5 | 76.0% | +129.2 | +3229 | 714 | 0.0 |
| echo_long | 5.7 | 88.5% | +96.0 | +2495 | 552 | 0.0 |
| **long+short (non-ov)** | **15.7** | 71.8% | +73.9 | **+5247** | 1160 | 0.0 |
| all_three (non-ov) | 15.7 | 71.8% | +73.9 | +5247 | 1160 | 0.0 |

> echo, core_long ile büyük oranda örtüşüyor (all_three = long_short ile aynı — echo yeni trade eklemiyor). SHORT_NOISY gerçek çeşitlendirme: +1000 TOTAL ekliyor.

### Karar / Yön

1. **Ana alfa = silence-LONG cascade reversion.** İyileştirme buraya odaklanmalı, narrow WR-pocket'lara değil.
2. **Doğru büyütme ekseni: eşiği düşür (100-150K) + silence + regime, hold 6h.** 40-61/ay, TOTAL +5000-8700. Mevcut ~7/ay live route bunun aşırı-filtrelenmiş küçük bir alt-kümesi.
3. **Şart: tail yönetimi.** Yüksek-frekans silence core'da tail_n 24-32. -150 stop + be_ratio<2 veto + US13-14 blok (tail'ler orada) ile tail kesilmeli; no-overlap/sizing gerçekçi modellenerek.
4. **SHORT_NOISY portföye gerçek değer katıyor** (+1000 TOTAL, düşük örtüşme); echo katmıyor (core ile örtüşüyor).
5. Sonraki test: silence core'u 100-200K'da tail-yönetimli + no-overlap + holdout ile gerçek-canlı frekansta doğrula (operator sign-off gerekli — bu live route'u materyal değiştirir).

### Guardrail
Research-only. Live executor / `.env` / order logic / leverage / sizing dokunulmadı.

---

## 30. Çeşitlendirme + Sinyal Keşfi + rv Stale Fix (2026-07-02)

Tam rapor: **`S34_SESSION_SONUC_RAPORU_2026-07-02.md`** (repo root).
Scriptler: `research_s34_divers_meta_sql / diversification_gauntlet / signal_discovery_v2 / refined_recipe_gauntlet / rv_stale_fix_validation / 100k_notmon_check`.

### 🔴 Pazartesi vetosu (en büyük yeni bulgu)
hour17 200K: Mon N=25 WR 32% avg −47.4 (TEST WR 25%); 100K: Mon N=27 WR 37% avg −42.4 (TEST WR 18.2%). İki evren + iki split tutarlı. `base_s2 + notMon` noov: WR 78.9% avg +79 TOT 3003 RA 17.7; TEST WR 90% mc 0.002. **hour17 live gate'inde Mon bloğu YOK — eklenmesi önerildi (operatör sign-off).**

### 🔴 vol_state ÖLÜ + rv düzeltmesi
`vol_state` producer'ı silinmiş; tablo 2026-06-05 19:00'da durdu. Sonrası 55 event'in hepsi bayat rv=0.253 ile rv-hit aldı (doğrusu 29 — %47 flip). **Edge artefakt DEĞİL:** mark-proxy rv (1m logret RMS 5m, eşik 0.0026337) ile s7≥3 TEST WR 86.4% avg +106 (bayat: 80/+86). Shadow runner'a `rv5m_robust()` eklendi (taze vol_state yoksa mark-proxy) — py_compile + `--once` ✓. **Restart operatör aksiyonu.** mark gap: 7 günde 11×>120s (max 786s) — izle.

### Çeşitlendirme kazananları
- **Interaction üçlüsü `rv+shelf+whale_lo`:** N=24 WR 91.7% avg +124 worst −35.4 **RA 59.6**; TEST WR 92.9. (Proxy-rv ile de: TEST WR 100 avg +122.)
- **Sizing = double-trigger** (rv+shelf→2u, +whale→3u): wTOT 5220 (flat 2586), perU +74.6, **mdd flat ile aynı (−237)**, RA 22.0; TEST RA 11.4 vs flat 2.4. 15x hesap-sim: 2.11x / −3.5% mdd. `unit=score` en çok para (9235) ama mdd −815.
- Premium sleeve s≥4/5: WR yüksek ama TOTAL düşürüyor (over-filter) — sadece sizing overlay olarak kullan.

### Yeni T0 sinyalleri (TEST-OOS geçti)
funding<0 (WR100 Δ+116), eth1h<−80 (WR100 Δ+107), tsl<115dk (WR81.8 Δ+84 N=22), two_sided BUY-liq≥68K (WR77.8 Δ+66, YENİ), basis>+0.9bp (WR86.7 Δ+62, YENİ). **score9** (s7+tsl+two_sided): s9≥6 TEST N=14 WR 92.9 avg +134.
❌ Kapananlar: gentleness (holdout tutarsız), taker1h, spread@T0, pt200 (non-monotonik, kırılgan).

### 100K frekans yolu güncel en iyi
`100K + hour17 + regime + notMon + s≥3`: full 22/ay WR 74% avg +81.6 TOT 8159; TEST WR 86.1 avg +95.9; noov 9.9/ay mc 0.0.

### Aksiyonlar
1. [URGENT süregelen] margin oversize. 2. [YENİ] Mon vetosu live gate (sign-off). 3. [YENİ] shadow runner restart (rv fix). 4. mark gap izle. 5. double-trigger sizing forward-shadow adayı.

### 30b. Shadow bucket implementasyonu + observer restart (2026-07-02)

Yapılan (observation-only, live executor koduna dokunulmadı):
- `tools/s34_realtime_shadow_runner.py` `compute_composite_score`: yeni T0 alanları loglanıyor — `rv_thr/rv_hit/shelf_hit/whale_hit`, `tsl_min/tsl_hit` (48h önceki-anchor reconstruct), `two_sided_usd/two_sided_hit`, `funding_rate`, `basis_bps`, `would_units` (double-trigger sizing gözlemcisi), `score10` (=score8 + tsl_hit + two_sided_hit). Eşikler: COMP_TSL_MIN=114.65 dk, COMP_TWO_SIDED=68,425 USD (SIGNAL_DISCOVERY_V2 TRAIN medyanları).
- `tools/s34_live_chart.py` 3 yeni bucket: `C_hour17_comp_notmon` (entry weekday≠Mon), `C_hour17_triple_rsw` (rv+shelf+whale hepsi hit), `C_hour17_score10` (score10≥6).
- Doğrulama: py_compile ✓, `--once` EXIT=0 ✓, compute_composite_score birim testi ✓ (proxy-rv yolu aktif, tüm alanlar dolu).
- Restart: shadow runner PID **12064** (yeni kod: rv fix + yeni alanlar), live chart PID **19640**. Dashboard `/api/data` 23 bucket döndürüyor, 3 yenisi mevcut. Eski duplicate shadow runner (22744, pid dosyası 5540 diyordu — yanlıştı) kapatıldı.

⚠️ **LIVE EXECUTOR ÖLÜ:** proses listesinde yok, pid dosyası 0, son log 2026-07-01 16:56 (hata yok, sessizce durmuş — muhtemelen §28b auth döngüsü/stop sonrası hiç kalkmamış). Yeniden başlatma = bilinçli arming adımı + sandbox dışı gerektirir → **operatör aksiyonu**: API key/IP whitelist çözüldüyse sandbox dışı `powershell -File start_eclipse.ps1`.

---

## 31. Mekanizma Araştırması Faz 1-2 Ön Sonuç (2026-07-02)

Yön değişimi: event-filtering → mechanism discovery + execution optimization.
Plan: `S34_MECHANISM_RESEARCH_PLAN.md` (5 faz). Store: `reports/research/s34/mechanism_store.sqlite` (418 event ≥100K + 418 saat-eşlenmiş kontrol, 11 Nis→şimdi book kapsamı, ~70 kolon: 5 pencere book/flow/impact, pull/refill, funding velocity, spot basis, 2s-15m exec grid). Scriptler: `s34_mechanism_feature_store.py`, `s34_mechanism_first_look.py`. Rapor: `S34_MECHANISM_FIRST_LOOK.md`.

**Veri denetimi:** open_interest 18 Nis'te ÖLÜ (OI tarihsel imkânsız → Faz 5 poller); spot_prices 5 Haz'da vol_state ile AYNI producer'la ölmüş (tarihsel basis var, forward yok); funding velocity mark_prices'tan tam ✓; book sadece L1.

**A — Mekanizma ayırıcıları (ungated ≥100K evren ~0 EV; TEST delta bps):** fund_rate<0 D=+115 (WR 69); px_rv hi +68; avg_trade_sz lo (retail) +67; fund_vel_1h lo +65; ofi_pre hi +61; two_sided hi +57; **bk_refill hi +31 (continuation %58.5→%45.2 — maker geri dönüşü = reversal MEKANİZMASI doğrulandı)**; **bk_pull hi +25 (likidite çekilmesi = continuation)**. fl_post5_impact TRAIN/TEST işaret çevirdi (robust değil).

**B — Pre-cascade imza (event vs kontrol sep):** px_rv 0.91 (vol cascade'den ÖNCE 2.2x yüksek); **basis_spot −15bps vs −4.4 (perp iskontosu = YENİ pre-cascade sinyali)** 0.79; ofi_pre satıcı 0.68; px_ret_1h −42 vs +2 0.67; cross-asset liq stress 1.0. Pre-cascade state detector kurulabilir görünüyor → Faz 3 tradeable EV testi şart.

**C — Giriş gecikme eğrisi (ungated):** tüm offsetler ~0 EV (ham cascade alfa değil — bilinen). refill-hi altkümede +10s avg +11.2 (mc 0.134). Saniye-hassasiyeti gate'siz evrende para kazandırmıyor; mekanizma+gate kombinasyonu Faz 2 devamında.

**Sıradaki:** Faz 2 tam taksonomi (ayırıcıları hour17 route üzerine bindir), Faz 3 pre-cascade EV, Faz 4 limit/VWAP fill + dinamik TP/stop, Faz 5 OI poller + spot producer canlandırma (operatör onayı).

### 31b. Mekanizma Araştırması Faz 2-5 TAMAMLANDI (2026-07-02)

**Faz 2 — Taksonomi** (`S34_MECHANISM_TAXONOMY.md`): Gated(hour17+regime, N=97) altkümede en güçlü ayırıcılar: **bk_pull hi D=+70.2 (WR 70.7%)**, retail sz lo +69.5, fund_vel_1h lo +57, fund_rate lo +53.2. **bk_refill gated'de işaret ÇEVİRDİ (−55.7)** → pull robust, refill değil. **Mekanizma kompoziti (0-7)**: gated mech≥4 N=60 WR 70% avg+49.9; TEST N=23 **WR 82.6% avg+88 mc 0.002**; mech≥5 TEST WR 81.2 avg+98.6. Funding 2x2: **SEVİYE > velocity** (rate_hi'de her iki vel hücresi negatif; rate_lo içinde vel farkı marjinal).

**Faz 3 — Pre-cascade** (`S34_PRECASCADE.md`): 20,831 örnek (5dk adım). Detektör GERÇEK: K≥4 → P(casc≤10m)=15.2% vs taban 7.2% (**lift 2.1x**). AMA para YOK: LONG@tetik avg −30.4 (mc 0.992 — düşüşü yiyor), SHORT@tetik avg −3.0 (mc 0.88 — düşüş fee'yi karşılamıyor). **❌ KAPANDI: erken giriş monetize edilemiyor; alfa post-cascade girişte kalıyor.**

**Faz 4 — Execution** (`S34_EXECUTION_OPT.md`, gated N=97): spread maliyeti ~0 (mark≈ask→bid — mark-bazlı backtest geçerli). **Limit giriş YİNE kaybediyor**: EV/sinyal market 30.9 vs limit10 24.2 / limit20 18.8 / limit30 15.0 (Q2 teyit, L1-fill ile). VWAP-5m ≈ market (fark yok). TP/SL taraması: TEST'te **baseline (TP yok, SL yok) hâlâ en iyi (avg 50.7)**; rv-ölçekli stop edge'i katlediyor (avg 3.8-2.8); fix300 stop de zarar (36.2). → Mevcut execution (T0 market, 6h, stop yok*) mekanik olarak DOĞRU. (*hour17 canlıda 300bps güvenlik stopu operatör kararı — ayrı konu.)

**Faz 5 — Veri altyapısı**: `data/oi_spot_poller.py` YAZILDI ve ÇALIŞIYOR (PID 17924, 60s, public endpoint, key gerekmez). open_interest + spot_prices tabloları tekrar doluyor (tick oi=3 spot=3 ✓). `start_eclipse.ps1`'e `oi_spot_poller` rolü, `stop_eclipse.ps1`'e pattern+pid eklendi — restart'larda kalıcı. Cross-exchange collector YAPILMADI (opsiyonel, operatör kararı).

**Net sentez:** Alfa = post-cascade reversion; mekanizması "likidite tutuldu (pull yok) + retail satışı + negatif funding + vol genişlemesi". Erken giriş kapalı, timing mikro-optimizasyonu kapalı, stop/TP/limit iyileştirmesi kapalı — kazanç kanalları: mekanizma-kompozit overlay (TEST WR 82.6) + conviction sizing + OI/basis forward verisi birikince yeni sinyaller.

---

## 32. Trade Management + Portföy Gauntlet (2026-07-02)

Rapor: `reports/research/s34/S34_TRADE_MGMT.md`. Script: `tools/research_s34_trade_mgmt_gauntlet.py`.
Evren: 98×200K(s≥2) + 90×100K(s≥3) → no-overlap 59 trade, 4.5 ay. Baseline 6h: avg +65.1, mdd −362.5 (TEST avg +58.3 mc 0.028). Kısıt: mdd artmayacak.

**✅ KAZANANLAR:**
- **M7 scale-in @−100bps (ilk 2h, +1 birim):** perU **+86.3** (baseline 65.1), wMDD **−329 (baseline'dan İYİ)**, worst aynı. Tek büyük kaldıraç — hem beklenti ↑ hem dd ↓. −75'te bozuluyor (mdd −510) → eşik spesifik −100. In-sample; forward + 15x sizing dikkat şart.
- **M5 profit-lock 200/100:** avg +67.2 (baseline 65.1), aynı mdd, WR 71.2. Marjinal ama bedava. ⚠️ Shadow'daki mevcut PROFIT_LOCK_100_50 gözlemcisi KÖTÜ varyantı ölçüyor (avg 47.7) — 200/100'e güncellenmeli.
- **M9 slot2 + günlük throttle −150bps:** N 59→95 (+61%), TOT 3839→**6488 (+69%)**, avg 68.3, WR 71.6; mdd −445 (slot1 −362'den kötü ama saf slot2 −616'dan çok iyi). Frekans istenirse en iyi paket; katı mdd kısıtında slot1 kalır.
- **M9 half-after-loss (savunma):** avg 53.4 (−18%) karşılığında mdd **−242 (−33%)** — dd-öncelikli operatör tercihi.

**❌ KAPANANLAR (hepsi baseline'dan kötü):**
- Partial exit'ler (½@+100/+150, ⅓'ler, ½@3h): avg 45-59, mdd iyileşmiyor.
- Loser time-stop (tüm grid): 53-55.
- Clock-exit (23/03/07 UTC): 39-42.
- Conviction-bazlı hold (M2/M8): hi VE lo için optimal 6h; TEST'te uniform=trainpick, conviction-yönetimi katkısız. 10-12h mdd'yi patlatıyor.
- **M10 hızlı timeframe trailing: 1m avg −1.5, 5m −0.1, 15m +8.9 — EDGE'İ YOK EDİYOR.** 1h-bar trailing 61.7 (yakın ama altında). Trade 1h+ ölçekte nefes alıyor; 15m altı yönetim zehir.

**M1 anatomi:** MFE +143.7 vs yakalanan +65.1 → giveback 73.6 (yarısı). hi-conv peak 249dk, lo-conv 148dk — ama kısa hold lo-conv'da bile kaybettiriyor (M2). Giveback'i kesmenin tek çalışan yolu lock 200/100 (marjinal).

**Net politika önerisi:** 6h uniform hold + erken çıkış yok + stop yok + (aday: scale-in −100 forward'da doğrula) + (opsiyon: lock 200/100) + (frekans istenirse slot2+throttle150). Shadow lock gözlemcisi 200/100'e çekilmeli.

### 32b. Yönetim gözlemcileri shadow'a eklendi (2026-07-02)

- `tools/s34_realtime_shadow_runner.py`: **SCALEIN_DIP100_2H_OBSERVER** (hour17 LONG route'ları; ilk 2h'de pnl≤−100 → paper +1 birim; kapanışta `combined_per_unit_net_bps` + `delta_vs_baseline_bps`; state'te `scalein_observer` agregi). **Profit-lock 100/50 → 200/100** güncellendi (M5: 100/50 kötü varyanttı). Order yok, observation-only.
- `tools/s34_live_chart.py`: bucket **`C_hour17_scalein100`** (scale-in tetiklenen trade alt-kümesi).
- Doğrulama: py_compile ✓, `--once` EXIT=0 ✓. Restart: shadow **PID 4708**, chart **PID 4748**; dashboard 25 bucket, yenisi görünür.
- ⚠️ Not: sandbox'tan başlatılan prosesler kalıcı olmayabiliyor (12064 sessizce öldü, err yok — shell child kill). Tüm roller (`oi_spot_poller` dahil) `start_eclipse.ps1`'e kayıtlı → **kalıcılık için sandbox dışı `start_eclipse.ps1` restart'ı yeterli** (idempotent, duplicate temizler).

---

## 33. AMI Faz 0-5 İnşası TAMAMLANDI (2026-07-02)

Whitepaper `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md` Appendix F brief'ine göre temel katmanlar inşa edildi. **Tamamen eklemeli** — live executor/.env/execution/risk/brain DOKUNULMADI; rollback = `ami/`, `data/ami/`, `docs/ami/` silmek.

### Paket: `ami/` (12 modül)
| Katman | Modül | Faz |
|---|---|---|
| Anayasa + enums | `constitution.py`, `enums.py` | 0 |
| Knowledge Graph + Audit + Failure Archive | `knowledge/objects.py`, `knowledge/store.py` → `data/ami/knowledge.sqlite` | 0 |
| **Epistemic Governor** (promotion/demotion gates, izinler, circuit breaker, belief revision, assumption cascade) | `governance/governor.py` | 5 |
| Multi-TF State Engine (1m..1W, veri-sağlığı yayılımı) | `states/engine.py`, `states/objects.py` | 1 |
| Structure transition matrix + TF conflict | `states/structure.py` | 2 |
| Trade Lifecycle + MFE classifier verisi | `lifecycle/engine.py` | 3 |
| Research OS (prereg-freeze ZORUNLU, marketplace 60/25/15) | `research/registry.py`, `research/marketplace.py` → `data/ami/research.sqlite` | 4 |
| DecisionTrace (immutable paket → `data/ami/decisions.jsonl`) | `decision/trace.py` | 5 |
| S34 tohumlama (9 KO + 12 mezarlık + 10 backlog sorusu) | `seed_s34.py` | — |

### Testler: 17/17 ✓
`tests/test_ami_knowledge_governance.py` (10) + `tests/test_ami_states_research.py` (7). Appendix F minimum seti dahil: provenance zorunlu, geçersiz promotion red, stale data applicability bloğu, çelişki izin düşürme, versiyon değişimi forward sıfırlama, research-only live yetki veremez + breaker/audit/assumption-cascade. (Not: pytest `--basetemp` scratchpad'e yönlendirilmeli — repo tmp izin sorunu.)

### Gerçek-veri doğrulama: 10/10 (`python -m ami.run_phase_checks` → `reports/research/s34/AMI_PHASE_VALIDATION.md`)
- Feed sağlığı: OI/spot HEALTHY (yeni poller ✓), vol_state doğru şekilde STALE raporlandı
- 11 state, TF conflict: 1m DOWN / 4h+1D UP (whitepaper'daki nested-structure örneği gerçek veride)
- Yön: LONG 0.58 → OPEN_LONG önerisi → governor **SHADOW_ONLY** (holdout-seviye bilgi live yetkisi alamadı — sınır çalışıyor)
- Lifecycle: 120 gerçek shadow trade replay; MFE+50 dağılımı: 59 continue / 13 negative / 17 time-pos
- Marketplace top: Q-MECHCOMP-FORWARD-001; deney E-MECHCOMP-FWD-001 donduruldu (hash 5bc8fd31…)
- Erken promotion `ConstitutionViolation` ile bloklandı

### Docs: `docs/ami/` — GAP_ANALYSIS, CHANGELOG (AMI-CHG-0001), ROADMAP, KNOWLEDGE_SCHEMA, STATE_TAXONOMY, RESEARCH_PROTOCOLS, DECISION_RECORDS/DR-0001

### Sonraki (roadmap): forward kanıt akışını shadow CLOSE'lara bağla → kalibrasyon (Brier) → S34_ALL.db toplu KO migrasyonu → Faz 6 latent states.

---

## 34. AMI Paket 1-3 + Faz Kapısı TAMAMLANDI (2026-07-02)

Raporlar: `AMI_FORWARD_EVIDENCE.md` · `AMI_MUTATION_REPORT.md` · `AMI_MFE50_EXPERIMENT.md` (hepsi reports/research/s34/). DR-0002, AMI-CHG-0002, whitepaper PATCH-0002 işlendi.

### Paket 1 — Automated Forward Evidence Pipeline ✅
`ami/research/forward_pipeline.py`: Shadow CLOSE → frozen spec eşleşmesi → versiyon/hash doğrulaması → EvidenceBundle → KO güncelleme → governor kararı → audit. Kurallar: R1 freeze-öncesi trade PRE_FREEZE reddi · R2 spec/candidate/dataset/execution değişimi=BINDING_INVALID (evidence taşınamaz) · R3 trade başına tek evidence (kalıcı PK, restart-güvenli) · R4 provenance zorunlu · R5 pipeline izin VERMEZ · R6 live dokunulmaz. **2 binding canlı:** E-HOUR17-FWD-001 (K-S34-HOUR17-001 ← LONG_HOUR17_HOLD6H) + E-CONVCOMP-FWD-001 (K-S34-MECH-COMPOSITE-001 ← LONG_HOUR17_COMPOSITE conv≥4); n=0'dan forward sayıyor, min_sample=20. E-MECHCOMP bilinçli BAĞLANMADI (shadow mech_score loglamıyor — dürüst eşleme). Koşum: `python -m ami.run_forward_pipeline` (idempotent).

### Paket 2 — Mutation/Adversarial Suite ✅ 20/20
`ami/mutation_suite.py` (tek kaynak) + `tests/test_ami_mutation_suite.py` + `python -m ami.run_mutation_report`. Yakalananlar: lookahead/freeze-öncesi, train-test leakage, stale-sensor-HEALTHY, research-only→LIVE, candidate-version+eski-evidence, dataset-hash, execution-model, post-hoc metrik değişimi, duplicate evidence, assumption-sonrası açık izin, çelişki-sonrası izin, restart duplicate, concurrent SQLite (WAL+busy_timeout eklendi), decision replay determinizmi, falsifiye→arşiv, provenance'sız KO, permission escalation, exploration→holdout, top3-removed eksik→promotion yok, DQ yayılımı. Toplam test: **17+20=37/37 ✓**.

### Paket 3 — İlk preregistered alpha deneyi: **FALSIFIES (dürüst null)** ✅
E-MFE50-001 (hash 4978edd7…, hesaplamadan ÖNCE donduruldu). Evren: 59 admitted → 42 milestone(+50); etiketler: 29 continuation / 5 negative / 8 stall. **TRAIN'de 10 available-at-+50 feature'ın hiçbiri cc≥0.85 kısıtını sağlayan kural veremedi (best=None)** → ayırma hipotezi TRAIN'de düştü. TEST kontrolleri teyit: HOLD cum=1313 (WR 84.6, PF 15.3, dd −46.9) > LOCK_ALL 1205 > EXIT_ALL 585. **HOLD baseline ayakta** (3. bağımsız doğrulama). Failure archive: NO_EDGE, retry=state-TRANSITION dizileri + çift-feature (YENİ prereg şartıyla). Evidence EV-MFE50-001 kayıtlı.

### Faz Kapısı durumu — GEÇİLDİ
| Kapı | Durum |
|---|---|
| Forward pipeline çalışıyor | ✅ software-correct + replay-validated (m01/05/06/09/12/15/18) |
| Mutation testler ihlalleri yakalıyor | ✅ 20/20 |
| Replay deterministik | ✅ m14 |
| Live sistem diff'i sıfır | ✅ git: executor/.env/execution/risk/brain değişmedi (shadow runner değişiklikleri önceki oturumlardan, observation-only) |
| İlk deney prereg + holdout ile tamam | ✅ (sonuç: FALSIFIES — dürüst) |
| Sonuç KO/failure sistemine işlendi | ✅ failure archive + EV-MFE50-001 |
| Baseline-vs-candidate raporu | ✅ AMI_MFE50_EXPERIMENT.md |
| Doğrulanmamış candidate'a operasyonel izin | ✅ YOK (hiçbir izin verilmedi, hiçbir bilgi terfi etmedi) |

**Faz 6-9'a geçiş artık serbest** (operatör kararıyla). Not: `docs/CLAUDE.md` operasyonel doktrini yüklendi — AMI işleri onunla uyumlu (execution/risk/brain'e sıfır dokunuş).

---

## 35. AMI Faz 6A — Latent State Discovery TAMAMLANDI (2026-07-02)

Rapor: `AMI_PHASE6A_LATENT.md`. Prereg E-LATENT6A-001 (hash a059e89d…, model ÖNCESİ frozen). DR-0003, AMI-CHG-0003, whitepaper PATCH-0003 işlendi.

**Paralel hat:** Forward pipeline kesintisiz — 2 binding VALID (E-HOUR17, E-CONVCOMP), n=0 birikimde, red/duplicate 0, versiyon bütünlüğü OK. **mech_score forward-only loglaması shadow'da CANLI** (PID 28080; mech_v1: 6 pre-bileşen T0'da + refill CLOSE'da; schema/feature version + provenance + missing-policy ile; geçmişe dönük evidence YASAK — bu tarihten önceki hiçbir trade E-MECHCOMP kanıtı sayılamaz).

**Faz 6A sonucu — dürüst REJECTED (geçerli null):**
- Dataset: 23,635 örnek (5m grid, 11 Nis→2 Tem), 9 backward-looking feature, outcome'suz (yapısal engel), missingness-mask + era-drop politikası.
- k-seçim (frozen kural): k=4, seed-ARI **0.851**, perturbasyon-ARI **0.991**, HMM çapraz-ARI 0.578; k=5/6 occupancy'den elendi.
- **Kronolojik validasyon DÜŞTÜ:** occupancy oranları [0.26, 2.08, **0.14**, **4.99**] — frozen bant [0.3,3.0] dışı; trans_corr 0.690 (geçti). Yorum: model instabilitesi değil **rejim kayması** (validasyon=Haziran rallisi: selloff-state LS-003 kayboldu, fiyat-düşüşsüz-stres LS-004 5× büyüdü). Kriter sonuca göre GEVŞETİLMEDİ → failure archive NO_STABLE_STATE (retry: daha uzun veri + rejim-koşullu kabul, YENİ prereg).
- State profilleri (nötr adlar): LS-001 sakin taban (215dk ort.), LS-002 çift-taraflı liq patlaması (buyliq 6σ), LS-003 keskin satış (%90 downtrend), LS-004 fiyatsız SELL-stres (5.9σ). Outcome eval (ayrı katman, bilgi amaçlı): ayrışma zayıf, LS-003 val'de n=4.
- Mutation: **15/15** (`tests/test_ami_latent_mutations.py`) — outcome/identity leakage, future-ts, norm-leakage, split overlap, missingness-fake-state, seed-instability→null yolu, permutation, artifact/version mismatch, stale-NaN, latent→LIVE yasağı, freeze-sonrası k/holdout değişimi. **Toplam AMI testi 52/52.**
- Geçiş kriterleri: hepsi ✓ (null geçerli sonuç; artifacts reproducible hash'li; live diff sıfır; hiçbir izin verilmedi; forward pipeline çalışır durumda).

Dürüst statüler: software-correct ✓ · replay-validated ✓ · latent-state **stable(exploration)/chronologically-unstable** · chronological-validation **failed** · alpha-incremental **non-incremental** · forward-validating ✗ · **operationally FORBIDDEN**.

Faz 6B/World Model'e GEÇİLMEDİ (operatör kararı bekler).

---

## 36. AMI Faz 6A-R — Regime-Conditioned + Drift TAMAMLANDI: PASS (2026-07-03)

Rapor: `AMI_PHASE6AR_REGIME.md` + `AMI_DRIFT_MONITOR.md`. Prereg E-LATENT6AR-001 (hash 1b6d0b2b…, sonuç öncesi frozen). DR-0004, CHG-0004, PATCH-0004 işlendi. Yeni dış veri YOK.

**Soru cevapları:**
1. 6A kırılması = **GERÇEK PİYASA REJİM KAYMASI** — rv30m/stress/buyliq/spread/trades PSI 0.16-7.4, missingness delta ~0 → DATA_ISSUE elendi; trend karışımı değişmedi (kayma yön değil likidasyon/vol YAPISINDA).
2. Rejim içi stabilite: 70/30 kronolojikte trend=UP, vol=LOW, vol=NORMAL, stress=NORMAL stabil; **walk-forward'da (4 fold) yalnız trend=UP PERSISTENT (band 3/4 + merkez-cos)**. trend=DOWN/RANGE, vol=HIGH, stress=STRESSED unstable.
3. DQ/missingness/session açıklaması ELENDİ (attribution tablosu raporda).
4. Transition matrix'ler rejim içinde tekrarlanabilir (trend=UP tc 0.98; boundary-guard'lı).
5-6. Incremental bilgi VAR ama sınıfı **market-description + RİSK** (untouched alpha eval: regime+latent PF 1.41, mdd −1363→−416) — **alpha NON-incremental** (top3-removed −458 = top-winner bağımlı; latent-only hepsi negatif). Promotion yok.

**Drift Monitor (research-only, öneri-only):** `python -m ami.latent.drift_monitor` — canlı durum **UNUSABLE** (stress PSI .55, spread 7.4, trades .89, occ-drift TV .51) → öneriler: applicability_restrict + shadow_suspend_suggest + retest + dq_investigation (governor'a; otomatik uygulanmadı). SHIFTED/UNUSABLE'da öneri listesi yapısal olarak boş olamaz.

**Test:** 14/14 rejim mutasyonu (`tests/test_ami_regime_mutations.py`) — outcome-rejim, val-threshold, fold/event leakage, dq-as-regime, missingness-as-state, tiny-regime, label-alignment, post-hoc gevşetme, transition-boundary, alarm-susturma, stale-artifact, version-mismatch, drift→LIVE yasağı. **Toplam AMI 66/66.**

**Paralel hat:** E-HOUR17 + E-CONVCOMP binding'leri VALID, accepted/rejected/dup=0 (hour17 penceresinde henüz CLOSE yok), min_sample 0/20; mech_score hattı canlı. Live diff SIFIR.

**Knowledge:** K-LATENT-REGIME-001 (HOLDOUT_VALIDATED, kapsam=trend/UP, max SHADOW; LIVE/SIZING/PORTFOLIO yasak; mekanizma adı YOK). Dürüst statüler: software-correct ✓ · drift-**attributed** ✓ · regime-conditioned **stable(trend=UP)** · walk-forward **passed(trend=UP)/failed(diğerleri)** · alpha **non-incremental** · forward-**not-validating** · **operationally forbidden**. Faz 6B'ye GEÇİLMEDİ.

---

## 37. AMI Faz 6A-R2 — Risk and Applicability Validation TAMAMLANDI: FALSIFIES / INSUFFICIENT_SAMPLE (2026-07-03)

Rapor: `AMI_PHASE6AR2_RISK.md/.json`. Prereg **E-RISKAPP-6AR2-001** (hash db07a737…,
hesaplamadan ÖNCE frozen; untouched-yokluğu PREREG'DE beyan edildi — 85-100% penceresi
6A-R hipotez kaynağı olduğundan CONTAMINATED, tam PASS bu dataset'te yapısal imkansızdı).
DR-0005, AMI-CHG-0005, whitepaper PATCH-0005 (0.2.4→0.2.5) işlendi. Yeni dış veri YOK.

**Soru:** regime(trend=UP)+latent-calm katmanı AYNI trade sayısı/maruziyet altında tail
risk'i azaltıyor mu, yoksa 6A-R'deki mdd −1363→−416 daralması az-trade-seçmenin mekanik
sonucu mu? (Yeni giriş alpha'sı ARANMADI.)

**Kurgu:** 328 no-overlap 6h LONG grid trade (mark-fill, FEE 5bps, MAE/MFE 5m çözünürlük);
veto yorumu (aynı popülasyonun altkümeleri); 4 expanding WF fold, per-fold DÜRÜST refit
(standardizer+rejim eşikleri+kmeans yalnız [0,val_lo), k=4/seed=11 sabit); kontroller:
matched-count moving-block bootstrap (2000, blok=5) + random-veto (2000) + regime-only +
latent-only; tam §4 risk metrik seti + loss-concentration; drift alarm lead/lag forward-only.

**Sonuç — dürüst null (frozen öncelik kuralıyla INSUFFICIENT_SAMPLE):**
- **Applicability çöküşü bulgunun kendisi:** aday veto fold 1-3'te n_cand=1/2/1 —
  "calm" state kimliği per-fold refit'te değişiyor (fold0 calm=1, sonra calm=0); 6A rejim
  kayması overlay'in SEÇİM kapasitesini öldürüyor. Toplam aday N=9 < 40, değerlendirilebilir
  fold 1 < 2.
- **Fold0 (tek değerlendirilebilir):** aday = regime-only BİREBİR AYNI set → latent
  incremental katkı SIFIR; cvar5 matched-pct 0.644 / random-veto-pct 0.650 (<0.75 eşik);
  retention_ratio 0.47 (<0.90) — kazananları orantısız atıyor. Random veto'dan ayırt edilemez.
- **6A-R N=14 hücresi (mdd −416) yeniden üretilemedi** per-fold dürüst artifact'larla —
  ALL-era-fit + hipotez-penceresi artefaktıydı. "Görünür risk azalması ≠ risk intelligence."
- **Drift alarmı SATÜRE:** validation erasında 13/13 pencere UNUSABLE, fp-suspension 0.69,
  lead teknik olarak ≥0 ama sürekli-açık alarm ayırt edici değil → applicability-degenerate;
  koruyucu applicability alpha İDDİA EDİLEMEZ. Ders: sabit-referans monitör rejim kayması
  sonrası bilgi taşımıyor; adaptif referans ayrı prereg ister.
- Failure archive: INSUFFICIENT_SAMPLE; retry = forward shadow ≥6 ay birikince YENİ prereg;
  kriter gevşetme YASAK.

**Mutation:** 13/13 (`tests/test_ami_risk_mutations.py`) — ham-MDD-cross-N, random-veto/
regime-only atlama, exposure-norm eksikliği, top-winner gizleme, post-hoc kriter (freeze),
retroaktif alarm, winner-sacrifice raporlamama, fold cherry-pick, UNUSABLE-drift'te
selection + stale artifact, LIVE/SIZING izin isteme, small-N bootstrap abartısı.
**Toplam AMI 79/79.**

**Paralel hat:** forward pipeline koşuldu — E-HOUR17-FWD-001 + E-CONVCOMP-FWD-001 binding
VALID, acc/rej/dup=0, n=0/20 (forward-validating İDDİASI YOK); mech_score hattı canlı.
**Live diff SIFIR** (executor/.env/execution/risk/brain dokunulmadı; DB salt-okunur).

Dürüst statüler: software-correct ✓ · frequency-normalized ✓ · risk-**non-incremental** ·
applicability-**degenerate/saturated** · walk-forward **failed** · forward-**not-validating** ·
**operationally FORBIDDEN**. Faz 6B/World Model/Digital Twin'e GEÇİLMEDİ.

---

## 38. C-BUY-FADE Yapısal + 8A Re-Entry Paketi TAMAMLANDI: FALSIFIES ×2 + silence-info (2026-07-03)

Raporlar: `BUYFADE_STRUCTURAL.md/.json` + `BUYFADE_REENTRY.md/.json` (reports/research/s34/).
Prereg'ler: **E-BUYFADE-STRUCT-001** (hash 70cf5acb…) + **E-BUYFADE-REENTRY-001** (hash
82a4e56b…) — her ikisi hesaplamadan ÖNCE frozen. DR-0006, AMI-CHG-0006, PATCH-0006
(0.2.5→0.2.6) işlendi. Route AYNEN korundu; live/shadow OTOMATİK DEĞİŞTİRİLMEDİ.

**Evren:** 391 ETH BUY cascade ≥200K event (veto: EUROPE 98, bear-squeeze 43, tail-7d 66);
train 234 / val 90 / untouched 54 (+24h purge, 13 purged). NOT_AVAILABLE (proxy'siz):
OI slope/accel, price-OI div, spot/perp participation, basis(kapsama<%60), depth>L1,
impact-per-$. p=0.010 denetimi: dashboard mc_p HARDCODED (100-perm tabanı); burada 20K perm.

**Ana bulgular:**
1. **Route ALL varyantı tarihsel replay'de NEGATİF:** −9.5/−1.1/−10.7 bps (tr/val/unt),
   PF 0.71, stop-rate %26. Shadow N=26 +2.8 GENELLEMİYOR.
2. **SILENCE tek gerçek bilgi ama T+30m-bilinir:** silence-subset +20.1/+30.4/+19.5 bps
   üç split tutarlı (WR 65-72); noisy −36/−26/−39. Matched-control (hour/size/rv/spread/
   trend/btc/density, 145 çift): diff **+54.4bps, p<5e-5** → proxy DEĞİL, bilgi GERÇEK.
   Kademeli (s30s→s10m) monotonik zayıflıyor (val +9→+19); breakdown median 4.3dk.
   Delayed-entry ile YAKALANAMIYOR (silence_3m_entry val +0.7) → **giriş alfası değil,
   geç-aşama yönetim/risk bilgisi.** Verdict: SILENCE_INCREMENTAL_INFO_T30M.
   (Not: verdict kodunda falsy-zero bug bulundu-düzeltildi; kriter değişmedi, rerun yapıldı.)
3. **Genesis/maturity (A):** frozen 6-sınıf; B_MATURE_TREND baskın (128) ve negatif;
   mature+silence hücresi güçlü (tr +31.7 n=73, val +32.8 n=23) ama silence-koşullu →
   T0 filtresi olarak **NON_PREDICTIVE**. Operatör hipotezi ("olgun LONG sonu") silence
   İÇİNDE doğru yönde, bağımsız değil.
4. **Timing (B):** 22 varyant (9 delay + 6 fiyat + 7 flow + 4 combo, hepsi frozen);
   train-best ofi_flip family_p=0.133 → **TIMING_NON_INCREMENTAL**. Val-parlak delay_600s
   (+10.5) untouched'ta −12.3 → split-instability, frozen protokol overfit'i yakaladı.
   Random-delay kontrolü −7.7 (p95 −4.1) — gecikme kendi başına para değil.
5. **Horizon (C):** 17 ufuk (5m..7D) HEPSİ unconditional negatif; event-high reclaim
   %62-93; path-sınıflar: DELAYED_FADE %35, FAILED_FADE %16, MULTI_HOUR %16.
   **SCALP_ONLY(default)** — continuation yapısal yok. H2 (4h-DOWN+silence) val h240
   +48 n=8 → INSUFFICIENT (yeni prereg adayı). H1/H3/H4 hücreleri boş/küçük.
6. **Multi-TF matrix (D):** 1W hep UNKNOWN (veri<20 hafta — dürüst); küçük hücreler
   INSUFFICIENT birleştirildi; hiçbir hücre train→val taşımadı.
7. **Management:** fixed 15m..24h, 4 time-stop (train-frozen eşik), 5 lock, 2 structural —
   hiçbiri fixed_45m'i val'de geçmedi → **MANAGEMENT_NON_INCREMENTAL**. SHORT→LONG
   transition (reclaim→4h LONG): tr +32/val +24 AMA top3_removed +2/−60 → top-winner
   bağımlı, yüksek standart geçilmedi (yalnız rapor).
8. **8A Re-entry:** S→S TÜM cooldown'larda train incremental NEGATİF (best cd=30m −709;
   val cap2 incr −138; random-timing pct 0.746<0.75; trigger dağılımı lower_high %65)
   → **SHORT_REENTRY_NON_INCREMENTAL (churn); H-RE-NULL doğrulandı.** cap99 kontrol
   kolu churn 1.22. S→L flip val n=2, L kolları eligible=0 → INSUFFICIENT.
   **Stop-taxonomy:** BAD_TIMING-stop sonrası re-entry train +19bps n=16 (vs
   WRONG_DIRECTION −21.5) — tek ilginç alt-sinyal, yeni prereg adayı.

**Kayıtlar:** EV-BUYFADE-STRUCT-001 + EV-BUYFADE-REENTRY-001 (FALSIFIES);
K-BUYFADE-SILENCE-INFO-001 (HOLDOUT_VALIDATED, max SHADOW; LIVE/SIZING/PORTFOLIO yasak);
failure archive += 5 (timing, genesis, management, S→S churn, flip/L INSUFFICIENT).
**Mutation:** 24/24 (`tests/test_buyfade_mutations.py`) — toplam suite 103/103.

**Operatör kararı bekleyen:** BUY_FADE shadow route'u (ALL tarihsel negatif) gözlemde
kalsın mı / silence-koşullu geç-aşama yönetim varyantına mı evrilsin? Otomatik değişiklik
YAPILMADI.

Dürüst statüler: software-correct ✓ (rerun ile) · chronological-validation **failed**
(silence-info hariç) · timing **non-incremental** · HTF **non-predictive** ·
**scalp-only** · management **non-incremental** · re-entry **churn** ·
forward-**not-validating** · **operationally forbidden**.

---

## 39. BUY-FADE Silence-Conditional Exit Timing TAMAMLANDI: REJECTED[econ] + T45_ROBUST (2026-07-03)

Rapor: `BUYFADE_SILENCE_EXIT.md/.json`. Prereg **E-BUYFADE-SILEXIT-001** (hash bd7d1f63…,
hesaplamadan önce frozen; untouched-kontaminasyonu beyanlı → tavan
CHRONOLOGICALLY_SUPPORTED_PENDING_FORWARD idi). DR-0007, AMI-CHG-0007, PATCH-0007
(0.2.6→0.2.7). Önceki iki BUYFADE deneyi DEĞİŞTİRİLMEDİ. Route/live/shadow DOKUNULMADI.

**Soru:** silence_v1 T+30m'de bilinir → T+45 sabit çıkış erken mi/geç mi/optimal mi?
(Yeni entry alpha ARANMADI; T0 entry/SL75/fee/universe sabit.)

**Kurgu:** Senaryo A (ANA) = T0 route + yalnız T30-açık pozisyonlara yönetim; survivor-safe
(pre-T30 SL'ler evrende: 7/0/1). Adaylar (frozen): 13 fixed (35m..24h) + 3 breakdown tanımı
× grace/pconf + 4 structural + 6 partial. Kontroller: noisy-aynı-çıkışlar, random-exit-timing
(2000), random-action (2000), Senaryo B (T+30 observer entry, ayrı etiketli).

**Sonuçlar (silence-open: train 99 / val 39 / untouched 22):**
1. **Kazanç T0→T30'da:** +36.7/+31.7/+22.9 brüt; T30→45 katkı +0.6/+3.0/+5.4.
   T+30 medyan unrealized +22bps — silence doğrulandığında hareket büyük ölçüde bitmiş.
2. **T45_EXIT_ROBUST:** hiçbir uzun fixed train+val'de tutarlı geçmedi (plato 45-120m,
   sonra düşüş; untouched 720/1440 sıçramaları val'de karşılıksız → gürültü).
3. **En iyi aday bd_first_buy50_g0m** (T30 sonrası ilk yeni BUY≥50K'da çık): 8/9 kontrol
   GEÇTİ (incr +8.4/+1.4/+10.1; random-p95 üstü; top3 OK; retention 0.74; tail OK) —
   **val econ +1.37 < 3bps → REJECTED** (kriter gevşetilmedi). Üstelik aynı çıkış
   noisy'de de val +31 → **silence'a özgü DEĞİL** (genel yönetim etkisi adayı).
4. **Senaryo B (T+30 observer entry): −15.9/−13.3** → silence girişi için ÇOK GEÇ;
   "silence = yönetim bilgisi, giriş sinyali değil" bulgusunun 3. bağımsız teyidi.
5. **Maturity (T+30-hesaplı):** immediate_noise_then_silent'ta 240m uzatma çöküyor
   (val +3 vs f45 +30); temiz-silence sınıfları 240m'de tutuyor (hücreler 10-18, iddia yok).
6. **Dar hipotez 4h-DOWN+silence:** val <12 → INSUFFICIENT_SAMPLE (gevşetilmedi).

**Kayıtlar:** EV-BUYFADE-SILEXIT-001 (FALSIFIES) + failure (NO_EDGE/econ; retry=forward
gözlemci + ≥6ay veri). **Mutation 16/16** (`tests/test_buyfade_silexit_mutations.py`) —
toplam suite **119/119**. 7 yeni survivor/lookahead guard'ı yeniden kullanılabilir.

**Operatör onayı bekleyen forward önerisi:** shadow'a observation-only `bd_first_buy50`
çıkış gözlemcisi (delta loglama; sipariş yok). Otomatik EKLENMEDİ.

Dürüst statüler: software-correct ✓ · survivor-bias-safe ✓ (audit tablolu) ·
chronological-validation: 8/9 geçti, econ'da düştü · profit-**pre-T30 baskın**
(frozen etiket POST_T30_CONTINUATION, oran ~10:1) · exit-timing **non-incremental** ·
regime **non-conditional (dar hücre INSUFFICIENT)** · forward-**not-validating** ·
**operationally FORBIDDEN**.

---

## 40. AMI×S34 MASTER PROTOKOL — PHASE 0 AUDIT TAMAMLANDI + MODEL HANDOFF (2026-07-03)

**Update (UTC):** 2026-07-03 · **Build:** BUILD-0001 · **Model:** FABLE 5 (Stage A) ·
**Active phase:** 0 → 1 hazır · **Protokol:** `docs/protocols/AMI_S34_MASTER_EXECUTION_PROTOCOL_v1.1.md`
(operatör master talimatı + token-verimli dokümantasyon politikası entegre).

### Yapılan (tamamı read-only; kod/runtime/store değişikliği SIFIR)
- 5 canonical belge tam reconcile edildi (whitepaper v0.3, reconstruction protocol,
  forward observatory, chart-native, reconciliation docx→text).
- Repo+runtime audit: 12 proses canlı (live executor KAPALI — beklenen); DB envanteri;
  AMI store'lar (11 KO / 21 failure / 2 forward binding VALID n=0/20); disk 982GB boş.
- **20 canonical artifact + 1 CSV üretildi** (repo kökü): LAST_VERIFIED_CHECKPOINT,
  REPOSITORY_RUNTIME_AUDIT, DOCUMENT_RECONCILIATION_MATRIX,
  CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER (10 conflict; 001-002 operatör kararlı),
  CURRENT_STATE_VS_TARGET_GAP_ANALYSIS (28 bileşen statülü), PROTECTED_COMPONENTS_MANIFEST,
  UNTOUCHED_LIVE_SHADOW_COMPONENTS, MASTER_ROADMAP (Phase 0-10 + faz eşlemesi),
  IMPLEMENTATION_DEPENDENCY_GRAPH, FIRST_SAFE_IMPLEMENTATION_BATCH (BATCH-P1-001),
  QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv (1058 satır; 192 verbatim Q867-1058, 866
  MISSING_CANONICAL_TEXT; üreteç `tools/ami_generate_question_matrix.py`),
  QUESTION_FAMILY_TO_ENGINE_MAP, WORKSTREAM_AND_EPIC_MAP, SCHEMA_AND_DATA_MIGRATION_MAP,
  HISTORICAL_RESEARCH_WAVES (W1-W12), FORWARD_OBSERVER_ROADMAP, TEST_AND_MUTATION_STRATEGY,
  STORAGE_COMPUTE_CAPACITY_PLAN, RISK_REGISTER (R-01..R-18), OPERATOR_DECISION_QUEUE
  (OD-001..OD-010), DEFINITION_OF_DONE_BY_PHASE, IMPLEMENTATION_PROGRESS_LEDGER,
  AMI_S34_BUILD_CHANGELOG, AMI_S34_RESEARCH_BACKLOG, FAILURE_ARCHIVE (index),
  CONTRADICTION_REGISTER (CT-001..004), SCHEMA_DICTIONARY, MIGRATION_LOG, REPRODUCIBILITY_REPORT,
  TEST_STATUS_LATEST.

### Kritik bulgular
1. **CONFLICT-001:** Protocol §16'nın "explicit Q396–Q730 registry (335 soru verbatim)"
   iddiasının karşılığı repoda YOK (research.sqlite'ta 14 slug-ID soru var) → OD-001.
2. **CONFLICT-002:** whitepaper v0.2 (PATCH zinciri 0.2.7) vs v0.3_COMPLETE ikiliği;
   Appendix H anlam çakışması → OD-002.
3. Canonical warehouse / artifact registry / contamination-exposure ledger / event-cycle
   identity MISSING — en erken eksik dependency = BATCH-P1-001 (warehouse iskeleti).
4. funding/oi history DB'leri Mayıs'tan bayat (OD-006); data/test_s34_* çöpü (OD-007);
   tüm canonical belgeler+ami/ untracked (R-16, OD-010).

### Durum satırları
- Running: 12 proses (PID'ler LAST_VERIFIED_CHECKPOINT.md'de) · live executor OFF.
- Forward: 2 binding VALID, raw n=0, independent-cycle n=0, min 20 · mech_score hattı canlı.
- Software test tabanı: 119/119 (§39) · Phase 0'da koşum yok (read-only).
- Live/shadow diff: SIFIR yeni değişiklik (shadow_runner'ın eski observation-only diff'i taban).
- Next safe dependency: BATCH-P1-001 · Bekleyen operatör kararları: OD-001..OD-010.
- **Next required model: SONNET 5** (operatör onayıyla).

---

## 41. BATCH-P1-001 DONE — canonical warehouse iskeleti (2026-07-03, Sonnet 5)

**Model:** SONNET 5 (operatör onayıyla geçildi) · **Active phase:** 1 · **Batch:** BATCH-P1-001 → DONE.

Operatör ek talimatları bu batch'e işlendi ve kalıcı: (1) **bucket freeze politikası** — Phase 0-5'te bucket koşulu optimize edilmez, yalnız inventory/verify/freeze/audit; material değişiklik = yeni version + forward N=0 (protokol §12.1). (2) **MD update politikası** — batch sonunda yalnız SYSTEM_STATE+LEDGER(+TEST_STATUS varsa); yeni MD istisna; bilgi kopyalanmaz (protokol §14 ek).

**Yapılan:** `ami/warehouse/` (schema.py + init_db.py + __init__.py, YENİ) + `data/ami/canonical.sqlite` (YENİ, M-0001) — 8 tablo (artifact_registry, artifact_lineage, question_families, question_registry, contradiction_registry, operator_decision_queue, namespace_registry, schema_versions). Hiçbir mevcut store/proses değişmedi.

**Testler:** 4/4 ✓ (`tests/test_ami_warehouse_schema.py` — idempotent init, round-trip, RO-open, version kaydı). `python -m ami.warehouse.init_db` 2× koşuldu, no-op doğrulandı. **Toplam AMI: 123/123.** Detay: TEST_STATUS_LATEST.md.

**Doğrulama:** protected diff yalnız pre-existing baseline gösterdi (`s34_realtime_shadow_runner.py`, bu oturumdan önceki); yeni dosyalar dışında git diff boş. Migration: MIGRATION_LOG.md M-0001 DONE. Schema: SCHEMA_DICTIONARY.md güncel.

**Sonraki:** BATCH-P1-002 (read-only artifact discovery → artifact_registry ingest, AMI-acronym collision gate) · BATCH-P1-003 (question matrix CSV + 14 slug-soru → question_registry seed). Model değişmedi, Sonnet 5 devam ediyor.

---

## 42. BATCH-P1-002 + BATCH-P1-003 DONE (2026-07-03, Sonnet 5)

**Active phase:** 1 · İkisi de operatör onayı gerektirmeyen read-only/reversible research-only schema hazırlığı kapsamında ardışık uygulandı.

**P1-002 (artifact ingest):** `ami/warehouse/artifact_ingest.py` — root *.md + docs/ami/** + docs/protocols/** → `artifact_registry` (64 kayıt: 62 CANONICAL + 2 UNDER_RECONCILIATION [whitepaper v0.2/v0.3, CONFLICT-002]). AMI-acronym collision gate: 0 quarantine. **Bulgu:** Reconstruction Protocol §2.2/15.2/18 kendi içinde repo'da bulunmayan harici bir "Advanced Metering Infrastructure" docx'ini quarantine etmeyi *anlatıyor*; ilk heuristik bunu yanlışlıkla protokolün kendisini quarantine ederek yakaladı (test kırmızı) → self-aware-AMI-doc ayrımı eklendi (`_AMI_SELF_ID_PATTERN`), düzeltildi ve regresyon testi eklendi. Harici docx repo'da aranıp bulunamadı — aksiyon gerekmiyor.

**P1-003 (question seed):** `ami/warehouse/question_seed.py` — CSV(1058) + research.sqlite legacy slug(14) → `question_registry`(1072) + `question_families`(29). No-fabrication doğrulandı (Q001 → MISSING_CANONICAL_TEXT + boş metin).

**Testler:** 6/6 + 5/5 ✓ (ayrı ayrı ve birlikte çalıştırıldı) · P1-001 4/4 regresyonsuz yeniden doğrulandı. **Toplam AMI: 134/134.** İki script de 2× koşuldu, idempotent (64/64, 1072/1072).

**Doğrulama:** protected diff yalnız pre-existing baseline (`s34_realtime_shadow_runner.py`) — bu batch'te dokunulmadı. Detay: SCHEMA_DICTIONARY.md, TEST_STATUS_LATEST.md, IMPLEMENTATION_PROGRESS_LEDGER.md.

**Sonraki:** BATCH-P1-004 (contradiction_registry + operator_decision_queue + artifact_lineage seed) veya Phase 2'ye geçiş (contamination/exposure ledger). Model değişmedi.

---

## 43. BATCH-P1-004 DONE — PHASE 1 TAMAMLANDI (2026-07-03, Sonnet 5)

**Active phase:** 1 → **KAPANDI**; sıradaki Phase 2.

**Yapılan:** `ami/warehouse/registry_seed.py` — mevcut MD kayıtlarını (yeniden yargılamadan, verbatim) warehouse'a yükledi: CONFLICT-001..010 + CT-001..004 → `contradiction_registry`(14); OD-001..010 → `operator_decision_queue`(10, hepsi OPEN — hiçbiri sessizce çözülmüş gösterilmedi); whitepaper v0.2↔v0.3 → `artifact_lineage`(1, relation=**UNDER_RECONCILIATION** — CONFLICT-002 kapanmadan SUPERSEDES gibi bir yargı fabrike edilmedi).

**Bulgu + düzeltme (regresyon sırasında yakalandı):** collision-gate P1-002'de düzeltilmişti ama SYSTEM_STATE.md'nin kendisi (Türkçe operasyon logu, "Artificial Market Intelligence" ifadesini hiç yazmıyor, yalnız "AMI"/"S34" kullanıyor) yeniden yanlış-pozitif quarantine aldı. `_AMI_SELF_ID_PATTERN`'e `S34`/`AMI×S34`/`AMI-S34` sinyalleri eklendi; regresyon testiyle kilitlendi (`test_collision_gate_self_aware_discussion_without_spelled_out_phrase`). 0 quarantine yeniden doğrulandı.

**Phase 1 DoD (DEFINITION_OF_DONE_BY_PHASE.md) — hepsi karşılandı:**
warehouse round-trip ✓ · **dump-roundtrip testi eklendi** (`.dump`→reload→tablo+veri doğrulaması, Protocol §23.1) ✓ · artifact registry 64 canonical dosyayı hash'li kapsıyor ✓ · question registry 1058 + 14 slug map ✓ · contradiction/OD/lineage seed ✓ · hiçbir mevcut store değişmedi ✓.

**Testler:** 7/7 (registry_seed) + 1 yeni dump-roundtrip testi. Tüm warehouse suite'leri son kez ikişerli gruplar halinde regresyonsuz doğrulandı (5+7=12, 8+5=13). **Toplam AMI: 144/144.** İki script 2× koşuldu (idempotent: 14/10/1).

**Doğrulama:** protected diff yalnız pre-existing baseline. Detay: SCHEMA_DICTIONARY.md (tüm tablo doluluğu güncel), TEST_STATUS_LATEST.md, IMPLEMENTATION_PROGRESS_LEDGER.md.

**Not (Opus kapısı):** Master protokole göre Opus REVIEW A, Phase 1**–3** kapanışında (yalnız Phase 1 değil). Phase 1 tek başına Opus onayı beklemeden Phase 2'ye geçebilir; REVIEW A, Phase 3 (event/cycle identity) tamamlandığında tetiklenecek.

**Sonraki:** Phase 2 — contamination ledger + researcher-exposure ledger + multiple-testing family registry + known-at kontrat modülü + funding/OI veri-kapsama denetimi (OD-006 girdisi). Model değişmedi, Sonnet 5 devam ediyor.

---

## 44. BATCH-P2-001..003 DONE — PHASE 2 TAMAMLANDI (2026-07-03, Sonnet 5)

**Active phase:** 2 → **KAPANDI**; sıradaki Phase 3 (event/cycle identity — Opus REVIEW A kapısı Phase 3 kapanışında).

**P2-001 (şema v1→v2, M-0002):** `ami/warehouse/schema.py` — 6 yeni tablo: `evidence_contamination` (§56.1), `researcher_exposure_ledger` (§56.4 + Protocol §7.2), `mt_family_registry` (§56.3), `causal_assumption_registry` (§56.5), `data_quality_events` (Protocol §7.1), `market_structure_versions` (Appendix N.11). Hepsi boş (Phase 6+'da beslenecek) — bu batch yalnız iskelet.

**P2-002 (known-at kontrat modülü):** `ami/timing/contract.py` — Observatory §6 (event_ts/available_at_ts/known_at_ts üçlüsü, `known_at_ts<=observer_trigger_ts` zorunlu kuralı → `LookaheadViolation(FUTURE_INFORMATION)`, PARTIAL_CANDLE reddi, DataQualityState enum'u [§6.4, MISSING sıfıra çevrilemez]). Tek kaynak — Phase 3+'taki tüm research engine'leri buradan import edecek (geçmiş script'lerin ad-hoc disiplini geriye dönük değiştirilmedi).

**P2-003 (OD-006 funding/OI denetimi — read-only, YENİ COLLECTOR YOK):** `ami/warehouse/funding_oi_audit.py` — 3 kaynağı (data/funding_history.db, data/oi_history.db, data/microstructure.db) read-only inceledi → `data_quality_events`(7). **Bulgu:** OI zaten CANLI (`oi_spot_poller`→microstructure.db; ETH ~3ay, BTC/SOL ~1.5gün — kısa pencere ayrı not edildi). **funding_rates'in HİÇBİR canlı üreticisi yok** — hem `data/funding_history.db` (son 2026-05-12, tek seferlik bulk backfill) hem `microstructure.db:funding_rates` (son 2026-04-13, yalnız ETHUSDT, kodda hiçbir INSERT bulunamadı — orphaned) donmuş durumda. **OD-006 bu bulgularla güncellendi, OPEN kaldı** — yeni funding collector aktivasyonu operatör kararı.

**Testler:** 6/6 (şema) + 10/10 (timing) + 4/4 (funding/OI audit) = 20 yeni test. Tüm warehouse suite'leri son kez ikişerli gruplar halinde regresyonsuz doğrulandı (14+12+14=40 toplam Phase1+2 testi). **Toplam AMI: 159/159.** Üç script de 2× koşuldu, idempotent.

**Doğrulama:** protected diff yalnız pre-existing baseline (`s34_realtime_shadow_runner.py`). Hiçbir proses başlatılmadı/durdurulmadı/config değişmedi. Detay: SCHEMA_DICTIONARY.md, TEST_STATUS_LATEST.md, IMPLEMENTATION_PROGRESS_LEDGER.md, MIGRATION_LOG.md (M-0002), OPERATOR_DECISION_QUEUE.md (OD-006 güncellendi).

**Sonraki:** Phase 3 — event/cycle identity (canonical cycle resolver, real-vs-proxy etiketleme; OD-003 onayı gerekir) + path/split-purge yardımcıları. Bu, Opus REVIEW A'yı tetikleyecek (Phase 1–3 kapanışı). Model değişmedi.

---

## 45. BATCH-P3-001..003 DONE — PHASE 3 TAMAMLANDI (OD-003 hariç) — OPUS REVIEW A CHECKPOINT (2026-07-03, Sonnet 5)

**Active phase:** 3 → KAPANDI (ami_cycles hariç, bilinçli). **Model değişim noktası: bu bölüm Opus 4.8'e geçiş checkpoint'idir.**

**Talimat uyumu:** Operatör "OD-003 konusunda material karar verme; seçenek+etki çıkar; onaydan bağımsız event identity/real-vs-proxy/immutable-ID/test-altyapısını tamamla; Phase 3 temizlenince Opus REVIEW A'da dur" dedi. Bu tam olarak uygulandı: **hiçbir cycle-definition/reset-censoring/direction-conflict seçimi yapılmadı**; `ami_cycles` 0 satırda bilinçli bırakıldı.

**P3-001 (event identity, schema v2→v3, M-0003):** `ami_events`/`ami_cycles`/`event_cycle_membership` şeması + `ami/identity/event_identity.py` (immutable `generate_event_id` — deterministik hash, aynı girdi=aynı ID, hiçbir yeniden-atama yok; `assert_not_pooled` — REAL_LIQUIDATION+PROXY_* karışımını R-09/CONFLICT-008 olarak kod-seviyesinde reddeder) + `ami/identity/shadow_ledger_ingest.py` (gerçek `reports/shadow/s34_state_machine_shadow.jsonl`'dan **251 gerçek event**, anchor-seviyesinde dedup — aynı anchor'a bağlı birden çok route tek event'e toplandı [235×1/15×2/1×4 route], 247 COMPLETED/4 RIGHT_CENSORED, hepsi source_quality=REAL_LIQUIDATION). İki gerçek bug regresyon sırasında yakalanıp düzeltildi: (1) bazı OPEN satırlarında `entry_ts_ms=None` (yalnız-izleme gözlemci satırları) — fabrikasyon yerine None bırakıldı; (2) aynı anchor'a birden fazla ledger "id" bağlanması ID çakışmasına yol açıyordu — anchor-seviyesi gruplama ile düzeltildi (Protocol §8.1'in tam kendisi, gerçek veride doğrulandı).

**P3-002 (cooldown sensitivity, §8.4):** `ami/identity/cooldown_sensitivity.py` — 6 pencere (1h/2h/4h/6h/12h/24h) → `event_cycle_membership`(1506 satır, **is_canonical=0 hepsinde**). Hiçbir pencere canonical seçilmedi.

**P3-003 (purge/embargo):** `ami/identity/split_utils.py` — cycle-grouped chronological split (whitepaper §51.1: aynı cycle train/val'e bölünemez) + purge (boundary-distance proxy) + embargo (tek-yönlü dead-zone). Jenerik — hangi `group_key_fn` verilirse (bugün: cooldown-sensitivity key; yarın: canonical cycle_id) onunla çalışır, OD-003'ten bağımsız.

**OD-003 (seçenek analizi, KARAR YOK):** OPERATOR_DECISION_QUEUE.md'de 3 boyut × seçenekleriyle genişletildi — (A) cycle-identity: 6h-proxy-v0 / Obs§5.4 çok-sinyalli resolver / Phase-6'ya erteleme; (B) reset/censoring: whitepaper 5-durum tam / mevcut 2-durum alt-kümesiyle kademeli; (C) direction-conflict: otomatik-öncelik-kuralı / WAIT+dashboard-flag. Hiçbiri önerilmedi/öncelenmedi.

**Testler:** 7/7 (şema, +ami_cycles-boş-testi) + 7/7 (event_identity) + 7/7 (shadow_ledger_ingest) + 5/5 (cooldown_sensitivity) + 8/8 (split_utils) = 34 test. **Toplam AMI: 187/187.** Tüm script'ler 2× koşuldu, idempotent (251/251 event, 1506/1506 membership).

**Doğrulama:** protected diff yalnız pre-existing baseline. Hiçbir proses/config değişmedi. Detay: SCHEMA_DICTIONARY.md, TEST_STATUS_LATEST.md, IMPLEMENTATION_PROGRESS_LEDGER.md, MIGRATION_LOG.md (M-0003), DEFINITION_OF_DONE_BY_PHASE.md (Phase 3 satırı OD-003 ayrımıyla güncellendi), OPERATOR_DECISION_QUEUE.md (OD-003 genişletildi).

```
==================================================
CRITICAL REVIEW CHECKPOINT
Completed model: SONNET 5
Next required model: OPUS 4.8

Review checkpoint: Opus REVIEW A (Phase 1–3 kapanışı, master protokol §3 Stage B)

Completed phases/batches: Phase 1 (BATCH-P1-001..004) + Phase 2 (BATCH-P2-001..003)
+ Phase 3 (BATCH-P3-001..003, ami_cycles hariç). Toplam 10 batch.

Changed files: ami/warehouse/{schema,init_db,artifact_ingest,question_seed,
registry_seed,funding_oi_audit}.py + ami/timing/{__init__,contract}.py +
ami/identity/{__init__,event_identity,shadow_ledger_ingest,
cooldown_sensitivity,split_utils}.py + data/ami/canonical.sqlite (yeni dosya)
+ 12 test dosyası + repo kökü dokümantasyon (SYSTEM_STATE, LEDGER,
TEST_STATUS, SCHEMA_DICTIONARY, MIGRATION_LOG, OPERATOR_DECISION_QUEUE,
DEFINITION_OF_DONE_BY_PHASE). Hiçbir mevcut store/proses/live/shadow
dosyası değiştirilmedi (protected diff: yalnız pre-existing
s34_realtime_shadow_runner.py baseline'ı).

Tests passed: 187/187 (119 taban + 68 yeni: 4+2 P1 warehouse şema/ingest/
question/registry testleri revize + P2 6+10+4 + P3 7+7+7+5+8). Tümü ayrı ayrı
ve ikişerli gruplar halinde sıralı çalıştırıldı (RAM kuralı).

Tests failed: 0 (yol boyunca 2 gerçek bug bulundu ve düzeltildi: collision-gate
self-aware-doc yanlış-pozitifi [iki kez, P1-002 ve P2 regresyonunda] ve
shadow-ledger anchor-dedup ID çakışması [P3-001] — ikisi de regresyon
testleriyle kilitlendi).

Scientific verdict changes: YOK. Hiçbir yeni alpha/hipotez test edilmedi;
bu üç faz tamamen altyapı (warehouse+identity+kontrat), araştırma değil.
Mevcut 21 failure-archive kaydı ve tüm KO'lar değişmedi.

Blocked questions: ami_cycles canonical seed → BLOCKED_PENDING_OPERATOR_
DECISION(OD-003). Q001-Q866 verbatim metin → OD-001. Whitepaper v0.2/v0.3
birleştirme stratejisi → OD-002. Toplam 10 açık operator decision
(OD-001..010), hiçbiri bu checkpoint'i bloklamıyor (yalnız ami_cycles'ı
bloklayan OD-003 hariç, o da açıkça işaretli).

Live/shadow diff: SIFIR. `tools/s34_state_machine_live_executor.py`, `.env`,
`execution/`, `risk/`, `brain/`, `tools/s34_realtime_shadow_runner.py`
(P0-öncesi baseline dışında) dokunulmadı. Hiçbir proses başlatılmadı/
durduruldu/restart edilmedi. Canlı 12 proses P0 audit'teki gibi.

Exact review scope for Opus 4.8:
1. CANONICAL RECONCILIATION (Phase 1): CANONICAL_PRECEDENCE_AND_CONFLICT_
   REGISTER.md'deki 10 conflict + warehouse'a doğru yansıdı mı (artifact_
   registry canonical_status, question_registry text_status/MISSING_
   CANONICAL_TEXT disiplini) denetle.
2. EVIDENCE/TIMESTAMP INTEGRITY (Phase 2): ami/timing/contract.py'nin
   Observatory §6 kontratını doğru uyguladığını (known_at<=trigger,
   partial-candle reddi) ve Phase 2 tablolarının (evidence_contamination vb.)
   şemasının whitepaper §56 ile birebir eşleştiğini doğrula.
3. EVENT/CYCLE/PATH FOUNDATION (Phase 3): immutable event_id şemasının
   gerçekten yeniden-atanamaz olduğunu, real-vs-proxy pooling guard'ının
   (assert_not_pooled) R-09/CONFLICT-008'i tam kapsadığını, ami_cycles'ın
   GERÇEKTEN boş ve hiçbir gizli cycle-definition varsayımı yapılmadığını,
   cooldown-sensitivity view'larının is_canonical=0 ile doğru işaretlendiğini
   ve split_utils.py'nin whitepaper §51.1 (aynı cycle train/val'e bölünemez)
   garantisini gerçekten sağladığını denetle.
4. HIDDEN DEPENDENCY INVERSION / DUPLICATE CANONICAL TRUTH: yeni warehouse'un
   mevcut data/ami/knowledge.sqlite + research.sqlite ile paralel ikinci bir
   truth yaratıp yaratmadığını kontrol et (tasarım niyeti: REFERANS, kopya
   değil — CONFLICT-005 kararı).
5. OD-003 seçenek analizinin (OPERATOR_DECISION_QUEUE.md) yanlı/eksik olup
   olmadığını, gerçekten hiçbir seçimin gizlice yapılmadığını doğrula.

Opus 4.8 remediation gerektirirse: Sonnet 5'e dönüş noktası — ilgili
BATCH-P3-00X'i düzelt, testleri tekrar koştur, bu checkpoint'i yeniden üret.
==================================================
```

Operatör Opus 4.8'e geçtiğini onaylamadan bir sonraki implementation adımına geçilmeyecek.

---

## 46. FABLE SUBSTITUTE REVIEW A — Phase 1–3 Gate İncelemesi (2026-07-03, Fable 5)

**Model kaydı: FABLE SUBSTITUTE REVIEW A** — Opus 4.8 model seçeneklerinde mevcut olmadığı için operatör talimatıyla Fable 5 substitute review yaptı. Bu bir Opus review DEĞİLDİR; Opus erişilebilir olduğunda operatör isterse aynı kapsam yeniden incelenebilir. Read-only: hiçbir kod/schema değişikliği yapılmadı.

### Bağımsız doğrulama (ingest kodundan bağımsız yeniden türetme)
Ledger ground-truth: 300 OPEN satırı → 18 rule_name=None (gözlemci-only, atlandı) → 269 logical trade (13 duplicate OPEN id, son-kayıt-kazanır; **kritik alanlar birebir aynı doğrulandı, 0 fark**) → **251 tekil anchor**. Warehouse: ami_events=251, sum(event_count)=269 — **birebir eşleşme**. Orphan membership 0; COMPLETED+NULL-end 0; kritik alan NULL'ları 0 (start_ts'de 2 NULL = giriş yapmamış izleme kayıtları, dürüst). Cooldown episode sayıları: 1h=151 / 2h=110 / 4h=88 / 6h=82 / 12h=55 / 24h=38 (4× yayılma — OD-003 denominatör seçiminin önemini gerçek veriyle gösteriyor).

### İnceleme alanları — sonuçlar
1. **Immutable event identity: SAĞLAM.** Deterministik SHA-256 hash (96-bit), yeniden-atama yolu yok, determinizm testli. F8 (LOW): ID, source_artifact_id'nin path-formuna bağlı — kod repo-relative normalize ediyor ama bunu kilitleyen açık test yok.
2. **Anchor-dedup + multi-route: DOĞRU.** 235×1 / 15×2 / 1×4 dağılımı bağımsız türetmeyle eşleşti; Protocol §8.1 ("ledger row ≠ independent event") gerçek veride uygulanmış. F2 (LOW): 13 duplicate-OPEN-id sessizce last-wins — bugün alanlar özdeş, ama davranış belgesiz/savunmasız.
3. **REAL/PROXY ayrımı: DOĞRU TASARIM, EKSİK ZORLAMA.** `assert_not_pooled` R-09'u kod olarak reddediyor ve testli; ANCAK henüz hiçbir aggregate yolu onu ÇAĞIRMIYOR — koruma opt-in durumda. F3 (MED): Phase 6 engine'leri population-assembly'yi bu guard'dan geçirmek ZORUNDA + mutation test.
4. **Missing timestamp handling: DÜRÜST.** entry_ts_ms=None → NULL (fabrikasyon yok); COMPLETED yalnız tüm bağlı trade'ler kapanınca; kısmi kapanış RIGHT_CENSORED + kısmi end-ts (belgeli).
5. **251 event ingestion: DOĞRULANDI** (yukarıdaki çapraz kontrol). F1 (MED): `event_family` alanı şu an route rule_name'i taşıyor (tek değer) — Protocol §7.4'ün kastettiği market-event taksonomisi değil. Phase 6 W1'den (cycle-integrity dalgası) ÖNCE haritalama stratejisi netleşmeli; DİKKAT: event_family event_id hash'ine giriyor → değişiklik yeni event_definition_version + yeni ID'ler demek, mevcut satırlar korunur (in-place mutasyon YASAK).
6. **Known-at/contamination kontratları: SAĞLAM İSKELET.** Obs §6 birebir (eşitlik sınırı dahil testli); Phase 2 tabloları whitepaper §56 alanlarıyla uyumlu; henüz tüketen yok (beklenen — Phase 4+/6'da bağlanacak).
7. **Cooldown sensitivity: KURALLARA UYGUN.** 6 pencere, hepsi is_canonical=0, deterministik content-derived key, hiçbir pencere gizlice canonical seçilmemiş.
8. **Purge/embargo: DOĞRU + SINIRLILIK BELGELİ.** Group-integrity garantisi testli; purge simetrik-mesafe proxy'si (per-record label-horizon-aware değil) — docstring'de açık; Phase 6 DoD'una "purge_seconds ≥ max outcome horizon" şartı taşınmalı (F7, INFO).
9. **Warehouse/lineage: TEK-TRUTH KORUNMUŞ.** Mevcut store'lar kopyalanmamış, referanslanmış (CONFLICT-005 kararına uygun); lineage'ta tek kayıt ve fabrikasyon yok. F4 (LOW): `connect()` PRAGMA foreign_keys açmıyor — tanımlı FK'lar SQLite'ta fiilen zorlanmıyor. F5 (LOW): schema_versions.note v3'te hâlâ "BATCH-P1-001 skeleton" yazıyor (kozmetik ama yanıltıcı). F6 (LOW): notional=max() aggregation semantiği schema dictionary'de belgesiz.
10. **OD-003 analizi: TARAFSIZ DOĞRULANDI** (hiçbir seçim gizlice yapılmamış; ami_cycles=0). **Öneri üretildi (karar operatörde): A2+B2+C2** — A1 Protocol §8.1'in açık metniyle çelişir; A3 Phase 6'yı denominatörsüz bırakır (episode yayılması 151→38 bunun kanıtı); B1'in dedektörleri henüz yok (B2 kademeli+reversible); C1 Protocol §8.3'ün açık uyarısını ihlal riski taşır. Detay OPERATOR_DECISION_QUEUE.md'de.

### Verdict: **PASS_WITH_REMEDIATION**

Blocker yok (ARCHITECTURAL/SCIENTIFIC/SAFETY sınıfında sıfır bulgu). Zorunlu remediation'lar — hepsi küçük, Phase 4 başlamadan tek batch'te:

| # | Severity | Dosya | Zorunlu düzeltme |
|---|---|---|---|
| F1 | MED | shadow_ledger_ingest.py + SCHEMA_DICTIONARY | event_family'nin route-scoped olduğunu belgele; Phase 6 W1 öncesi taksonomi-haritalama kararını OD/W1-prereg'e bağla (ID-mutasyon yasağıyla) |
| F2 | LOW | shadow_ledger_ingest.py | duplicate-OPEN last-wins davranışını belgele + kritik-alan-eşitliği savunma kontrolü ekle |
| F3 | MED | (Phase 6 şartı) | population-assembly helper'ı assert_not_pooled'u zorunlu çağırmalı + mutation test — TEST_AND_MUTATION_STRATEGY kapsamında Phase 6 DoD'una bağlandı |
| F4 | LOW | schema.py connect() | PRAGMA foreign_keys=ON |
| F5 | LOW | schema.py init_schema() | schema_versions.note güncel versiyonu yansıtsın |
| F6 | LOW | SCHEMA_DICTIONARY.md | notional=max() semantiği tek satır belge |

**Sonnet'in sonraki exact batch'i: BATCH-P3-004 (remediation)** — F1/F2/F4/F5/F6 (F3 Phase-6-DoD notu olarak; F8'e ID-path-form kilitleme testi dahil edilebilir). Küçük, reversible, new-file-yok (mevcut ami/ dosyalarına ek + 1-2 test güncellemesi). Sonrasında: OD-003 onaylanırsa BATCH-P3-005 (canonical A2 resolver); onay gelmezse Phase 4 BATCH-P4-001 (candle normalizasyonu) — ikisi de birbirini bloklamaz.

```
==================================================
REVIEW COMPLETE
Completed model: FABLE 5 (SUBSTITUTE — Opus 4.8 unavailable)
Next required model: SONNET 5
Review verdict: PASS_WITH_REMEDIATION
Accepted components: warehouse v3 (14 tablo) · artifact/question/registry
  seed'leri · timing kontratı · event identity + 251-event ingest ·
  cooldown sensitivity · split/purge/embargo · OD-003 seçenek analizi
Blocking findings: YOK
Required remediation: F1,F2,F4,F5,F6 (BATCH-P3-004) + F3 Phase-6-DoD şartı
Required new tests: duplicate-OPEN eşitlik kontrolü · ID path-form kilidi ·
  FK-enforcement smoke · (Phase 6'da) pooled-population mutation testi
Operator decisions: OD-003 önerisi A2+B2+C2 — ONAY BEKLİYOR; OD-001..010 açık
Exact resume instruction for Sonnet 5: "SYSTEM_STATE §46'yı oku;
  BATCH-P3-004 remediation'ı uygula (F1/F2/F4/F5/F6); testleri koştur;
  protected diff doğrula; sonra OD-003 durumuna göre P3-005 veya P4-001."
==================================================
```

---

## 47. BATCH-P3-004 + BATCH-P3-005 DONE — PHASE 3 TAMAMEN KAPANDI (2026-07-03, Sonnet 5)

**Active phase:** 3 → **TAMAMEN KAPANDI** (OD-003 dahil, hiçbir istisna kalmadı). **Sonraki: Phase 4** (chart-native object foundation).

### BATCH-P3-004 — FABLE SUBSTITUTE REVIEW A remediation'ları

F1 (event_family route-scoped): `shadow_ledger_ingest.py` docstring'e belgelendi + `event_id` hash'inin `event_family` değişince otomatik farklı ID ürettiğini kanıtlayan test eklendi (in-place mutasyon yapısal olarak imkansız). F2 (duplicate-OPEN last-wins): `DuplicateOpenConflict` eklendi — yalnız kimlik alanları (anchor_ts_ms/rule_name/signal) kontrol edilir; **gerçek veri incelemesi entry_ts_ms'in gecikmeli-giriş gözlemcilerinde meşru None→değer geçişi yaptığını ortaya çıkardı** (kod buna göre düzeltildi, kimlik alanlarında 20 duplicate'in hiçbirinde divergence yok — bağımsız doğrulandı). F4 (`PRAGMA foreign_keys=ON`): eklendi. F5 (schema_versions.note dinamik): düzeltildi.

**Kritik yan bulgu:** F4'ü açmak, `question_seed.py`'de BATCH-P1-003'ten kalan gerçek bir bug'ı ortaya çıkardı — `question_registry.family_id` UPSERT'i `question_families.family_id` ile eşleşmiyordu (prefix farkı: "FAM_X" vs "X"), **1058/1058 satır etkilenmiş**, sessizce (FK kapalıyken hiç hata vermeden). Kod düzeltildi (doğru değer + UPDATE SET'e eklendi) ve gerçek DB onarıldı (repair sonrası 0 mismatch, bağımsız doğrulandı). **Bu, review disiplininin (FK enforcement + bağımsız çapraz-kontrol) canlı veri bütünlüğünü koruduğu somut bir örnek.**

### BATCH-P3-005 — Canonical cycle resolver (OD-003: operatör A2+B2+C2'yi onayladı)

`ami/identity/cycle_resolver.py` — `cycle_definition_version="canonical-v1"` **immutable** olarak dondu:
- **A2:** symbol+family grouping + 4h continuity gate (mevcut `LONG_HORIZON_MS` operasyonel sabitine dayalı, keyfi değil) + `ami/states/engine.py`'den point-in-time dominant-structural-state (1h TF, known-at-safe `ts_ms<=?` sorgularla — 21ms/çağrı, 252 event ≈80sn). Cascade-continuity/shared-parent-event/observer-horizon sinyalleri **NOT_IMPLEMENTED** — belgeli sınırlılık (veri modeli henüz yok), fabrikasyon değil.
- **B2:** 2-durum censoring (COMPLETED yalnız tüm cycle üyeleri kapandıysa).
- **C2:** direction_conflict flag-only — otomatik çözülmez, WAIT-eşdeğeri.

**Sonuç (gerçek veri, bağımsız doğrulandı):** 252 event → **167 cycle** (162 COMPLETED/5 RIGHT_CENSORED, 15 direction_conflict=1). confidence ortalaması 1.0 (tüm state lookup'ları başarılı). sum(event_count)=252=ami_events sayısı ✓. Sensitivity view'lar (6 pencere, 1512 satır, is_canonical=0) **hiç dokunulmadı** — append-only olarak yalnız 252 yeni is_canonical=1 satırı eklendi. 0 orphan membership.

### Testler ve doğrulama
34 yeni/güncellenen test (9+8+5+7+10+4+7+10+5+8+11 — dosya bazında TEST_STATUS_LATEST.md'de). **Toplam AMI: 203/203 ✓.** Tüm suite'ler ikişerli gruplar halinde sıralı çalıştırıldı. Tüm script'ler 2× koşuldu (idempotent — ledger canlı process'ten büyüdüğü için 251→252 event doğal artış, hata değil). Protected diff: yalnız pre-existing baseline.

### Kalıcı kayıtlar
- `OPERATOR_DECISION_QUEUE.md`: OD-003 → **IMPLEMENTED** (A2+B2+C2, canonical-v1 dondu). Diğer 9 karar hâlâ OPEN (OD-001/002/004-010).
- `MIGRATION_LOG.md`: M-0004 (FK+note+family_id repair), M-0005 (canonical-v1 seed).
- `SCHEMA_DICTIONARY.md`: ami_cycles=167, event_cycle_membership=1512+252, question_registry repair notu.

### Phase 4 başlangıç durumu
**Hazır.** Phase 3'ün tüm DoD kriterleri karşılandı (event/cycle identity versiyonlu VE onaylı, real-vs-proxy zorunlu, raw-N/independent-cycle-N ayrı — ami_cycles.event_count ile ami_events sayısı). Phase 4 (candle/swing/level/push, closed-candle-only, known_at kontratlı) `ami/timing/contract.py`'yi doğrudan kullanabilir; hiçbir observer aktive edilmeyecek. Opus/Fable-substitute REVIEW B kapısı Phase 4-5 kapanışında. Model değişmedi, Sonnet 5 devam ediyor — operatör onayı beklemeden Phase 4 BATCH-P4-001'e başlanabilir (protokol §13: bu tür research-only schema/kod hazırlığı operatör onayı gerektirmez).

---

## 48. BATCH-P4-001..004 + P5-001 DONE — PHASE 4-5 TAMAMLANDI — REVIEW B CHECKPOINT (2026-07-03, Sonnet 5)

**Active phase:** 4-5 → **KAPANDI**. Sonraki: Phase 6 (historical research waves) — yalnız REVIEW B onayı sonrası.

### Phase 4 — Chart-native object foundation (BATCH-P4-001..004)

Schema v3→v4 (M-0006): `ami_candles`, `ami_candle_morphology`, `ami_swings`, `ami_levels`, `ami_pushes`. Kod: `ami/chart/{candle_builder,candle_morphology,swing_extractor,level_registry,push_geometry}.py`. Tüm veri **gerçek** (`data/microstructure.db:agg_trades`'ten, ETHUSDT, 48h lookback) — sentetik değil.

- **Candle (§4.1/§6.1):** closed-candle-only (known_at_ts=close_ts_ms; still-forming bucket asla saklanmaz; boş bucket fabrikasyon yerine atlanır) → **3456 candle** (2880×1m+576×5m), 0 OHLC ihlali. Morfoloji: §6.1 atomik özellikler + 3-durum close-quality (provisional eşik, belgeli). REJECTION/ACCEPTANCE/FOLLOW_THROUGH NOT_IMPLEMENTED.
- **Swing (§4.2):** N=3 simetrik fraktal, `known_at_ts` (confirming candle) her zaman `pivot_ts`'ten sonra (testli, gerçek veride 0 ihlal) → **556 swing** (294 HIGH/262 LOW). Flat-top tie'lar fabrikasyon yerine atlanır.
- **Level (§4.3):** SESSION_HIGH/LOW + PREVIOUS_DAY_HIGH/LOW (session sınırları `s34_state_machine_live_executor.py`'den **read-only referans**, import edilmedi) + SWING_HIGH/LOW → **576 level**, touch/rejection/acceptance istatistikleriyle. 12 level_type NOT_IMPLEMENTED (volume-profile/breakout altyapısı yok).
- **Push (§7.1):** ardışık alternating swing çiftleri → **454 push** (228 UP/226 DOWN). **2 gerçek bug bulunup düzeltildi** (yol boyunca, kod yazarken): (1) path_length hareketli-taban bps ile toplanıyordu, displacement'la karşılaştırılamaz hale geliyordu; (2) path yalnızca pencere-içi mum close'larına bağlanıyordu, swing'in gerçek pivot fiyatına (high/low) uzanan uç segmentleri atlıyordu. İkisi de `efficiency_ratio>1` (üçgen eşitsizliği ihlali, matematiksel olarak imkansız) ile yakalandı; düzeltildi, gerçek DB onarıldı (369 ihlalli satır → 0), kalıcı sınır-testi eklendi (`test_efficiency_ratio_never_exceeds_one`).

### Phase 5 — Feature dictionary + duplicate-engine denetimi (BATCH-P5-001)

Feature dictionary SCHEMA_DICTIONARY.md'de konsolide edildi (5 tablo = AMI'nin tek canonical chart-feature kaynağı). Duplicate-engine denetimi: `tools/s34_v_engine_failure_anatomy.py:candle_features()` farklı-normalizasyonlu (ref_price bps, CN'nin range-oranı değil), belirli bir tarihsel araştırmaya (FADE_DIRECTION) bağlı bir mum-morfolojisi hesaplayıcısı bulundu. **Karar: dokunulmadı** (geçmiş araştırma scriptleri yeniden yazılmaz). Kural: Phase 6+ yeni chart-feature ihtiyacı `ami/chart/*` kullanır. known-at testleri 4 nesnenin tamamında doğrulandı.

### Testler
32 yeni test (7+6+6+6+7). **Toplam AMI: 235/235 ✓.** Tüm suite'ler ikişerli gruplar halinde sıralı çalıştırıldı, tüm script'ler 2× koşuldu (idempotent). Protected diff: yalnız pre-existing baseline.

```
==================================================
CRITICAL REVIEW CHECKPOINT
Completed model: SONNET 5
Next required model: OPUS 4.8 (veya mevcut değilse FABLE SUBSTITUTE REVIEW B)

Review checkpoint: REVIEW B (Phase 4–5 kapanışı, master protokol §3 Stage B)

Completed phases/batches: Phase 4 (BATCH-P4-001..004) + Phase 5 (BATCH-P5-001).
Toplam 5 batch, Phase 1-3 üstüne (23 batch toplamda).

Changed files: ami/chart/{__init__,candle_builder,candle_morphology,
swing_extractor,level_registry,push_geometry}.py + ami/warehouse/schema.py
(v4: ami_candles/ami_candle_morphology/ami_swings/ami_levels/ami_pushes) +
data/ami/canonical.sqlite + 5 yeni test dosyası + repo kökü dokümantasyon
(SYSTEM_STATE, LEDGER, TEST_STATUS, SCHEMA_DICTIONARY, MIGRATION_LOG).
Hiçbir mevcut store/proses/live/shadow dosyası değiştirilmedi (protected
diff: yalnız pre-existing s34_realtime_shadow_runner.py baseline'ı).
tools/s34_state_machine_live_executor.py YALNIZ READ-ONLY okundu (session
boundary convention'ı almak için), hiç yazılmadı/import edilmedi.

Tests passed: 235/235 (203 Phase1-3 taban + 32 yeni Phase4-5). Tümü ayrı
ayrı ve ikişerli gruplar halinde sıralı çalıştırıldı (RAM kuralı).

Tests failed: 0 (yol boyunca 2 gerçek bug bulundu ve düzeltildi — push
geometrisindeki path-length hesap hataları; ikisi de matematiksel
imkansızlık [efficiency_ratio>1] ile yakalandı, gerçek DB onarıldı,
kalıcı sınır-testiyle kilitlendi).

Scientific verdict changes: YOK. Hiçbir yeni alpha/hipotez test edilmedi;
Phase 4-5 tamamen altyapı (chart-native nesneler + feature dictionary),
araştırma değil. Mevcut failure-archive ve KO'lar değişmedi.

Blocked questions: Chart-native question aileleri (Q867-1058) hâlâ
FUTURE_PHASE — Phase 6+ tarihsel dalgalarını bekliyor. OD-001/002/004-010
hâlâ OPEN, bu checkpoint'i bloklamıyor.

Live/shadow diff: SIFIR. Hiçbir proses başlatılmadı/durduruldu/restart
edilmedi. Canlı 12 proses (+shadow ledger'ın kendi büyümesi, dış süreç)
değişmedi.

Exact review scope for Opus 4.8 (veya substitute):
1. CHART OBJECT FOUNDATION (§4): candle/swing/level/push şemalarının CN
   spec'iyle uyumunu, known_at_ts disiplininin her nesnede GERÇEKTEN
   uygulandığını (closed-candle-only, pivot≠known_at, session/day
   fully-elapsed kuralı, push pullback ileri-taşıma) doğrula.
2. SHARED FEATURE ENGINES / DUPLICATE PREVENTION (Phase 5 DoD): duplicate-
   engine denetiminin yeterli olup olmadığını (yalnız 1 tarihsel örnek
   bulundu, repo genelinde başka taranmamış alan var mı) değerlendir.
3. PUSH GEOMETRİ DÜZELTMESİ: efficiency_ratio hesabının artık üçgen
   eşitsizliğini GERÇEKTEN sağladığını (kod + test + gerçek veri) bağımsız
   doğrula — bu, review sürecinin canlı veri hatası yakaladığı 2. örnek
   (1.'si BATCH-P3-004'teki family_id FK bug'ıydı).
4. NOT_IMPLEMENTED alanların (REJECTION/ACCEPTANCE close-quality, 12
   level_type, liquidation_notional, cascade-continuity) gerçekten
   fabrike edilmediğini, yalnız NULL/eksik olarak dürüstçe işaretlendiğini
   doğrula.
5. LIVE EXECUTOR READ-ONLY ERİŞİM: `tools/s34_state_machine_live_executor.py`
   okumasının (session_label mantığı için) gerçekten yalnız-okuma kaldığını,
   hiçbir import/coupling oluşmadığını doğrula.

Opus 4.8 (veya substitute) remediation gerektirirse: Sonnet 5'e dönüş
noktası — ilgili BATCH-P4-00X/P5-001'i düzelt, testleri tekrar koştur, bu
checkpoint'i yeniden üret.
==================================================
```

Operatör Opus 4.8'e (veya erişilemezse Fable substitute'a) geçtiğini onaylamadan bir sonraki implementation adımına (Phase 6) geçilmeyecek.

---

## 49. FABLE SUBSTITUTE REVIEW B — Phase 4–5 Gate İncelemesi (2026-07-03, Fable 5)

**Model kaydı: FABLE SUBSTITUTE REVIEW B** — Opus 4.8 erişilebilir olmadığı için operatör talimatıyla Fable 5 substitute review. Bu bir Opus review DEĞİLDİR. Read-only: hiçbir kod/schema/veri değişikliği yapılmadı (bulunan kusurların onarımı Sonnet batch'ine bırakıldı — model rol ayrımı).

### Bağımsız doğrulama (SQL çapraz-kontrol, gerçek warehouse verisi)
**Geçen invariant'lar:** 3456 candle 0 OHLC ihlali · 0 PARTIAL satır · known_at==close_ts 3456/3456 · taker buy+sell≤volume ✓ · morfoloji 1:1 + label/clv tutarlılığı 0 mismatch · 556 swing'de known_at>pivot_ts 0 ihlal + tümü uniform 4-bar confirmation + pivot_price kaynak mumla birebir · 454 push'ta efficiency>1 = 0, yön/işaret mismatch 0, FK orphan 0, push.known_at ≥ end-swing.known_at ✓ · **push onarımının doğruluğu teyit** (min/ort/maks eff = 0.14/0.86/1.00 — matematiksel olarak tutarlı aralık) · duplicate engine (`s34_v_engine_failure_anatomy.py`) dokunulmamış + `ami/chart`'a 0 coupling.

### Bulgular

| # | Severity | Durum | Bulgu |
|---|---|---|---|
| **F-B4** | **MED-HIGH** | **MANIFEST — depoda 4 hatalı satır** | Lookback penceresinin başında KESİLEN session/day dönemleri truncated veriden level üretmiş: 2026-07-01 US session'ı 8 saatin yalnız 1'iyle (20:00-21:00), 2026-07-01 günü 24 saatin yalnız 4'üyle SESSION_/PREVIOUS_DAY_HIGH/LOW üretti — bu değerler dönemin gerçek high/low'u DEĞİL. Fix: origin'i ilk-mevcut-mumdan önce olan dönemleri atla (veya DATA_TRUNCATED etiketle) + mevcut 4 satırı temizle/yeniden üret + pencere-kenarı testi |
| **F-B1** | MED | LATENT (mevcut veri tetiklemiyor) | `level_registry`'de session/day `known_at_ts = son üye mumun close'u` — sınırdan ÖNCE olabilir (sessiz kuyruk/gap durumunda "dönem high'ı" dönem bitmeden bilinir görünür = lookahead). Mevcut ETHUSDT 1m verisi her dakika mumlu olduğundan 0/20 tetikleniyor ama kod yolu kusurlu. Fix: `known_at = max(son close, dönem-sonu sınırı)` + sınır testi |
| **F-B2** | MED | KULLANIM KURALI | `ami_levels.touch/rejection/acceptance_count` build-anı aggregate'leri — point-in-time DEĞİL. Phase 6 bir engine bunları anchor-anında feature olarak okursa lookahead olur. Fix: SCHEMA_DICTIONARY'ye açık uyarı + Phase 6 engine gateway'i bu kolonları as-of-recompute olmadan tüketemesin (mutation test) |
| **F-B3** | LOW-MED | DÜRÜSTLÜK | `ami_candles.data_quality` her satırda AVAILABLE; oysa `microstructure.db:gaps` tablosunda 812 collector-gap kaydı var — boş bucket "trade yok" ile "collector kesintisi"ni ayırt etmiyor. Fix: gaps tablosuyla çapraz-işaretleme (GAPPED etiketi), Phase 6 öncesi |
| **F-B5** | MED | ENFORCEMENT EKSİK | "Phase 6 yalnız ami/chart/* kullanır" + `assert_not_pooled` zorunluluğu şu an yalnız dokümantasyon. Fix: P6-001'de population/feature erişimi TEK gateway modülünden geçsin (chart-feature import + pooling guard + contamination ledger kaydı zorunlu) + mutation testleri — Review A'daki F3 ile birleşik |

Ayrıca doğrulanan tasarım artıları: swing N-bar penceresinin pozisyonel (zaman değil) olduğu belgeli; flat-top tie'ların atlanması fabrikasyon-yasağına uygun; push known_at ileri-taşıması muhafazakâr-güvenli yönde; NOT_IMPLEMENTED alanların tümü NULL/belgeli (fabrikasyon yok); feature versioning 5 nesnede de mevcut.

### Verdict: **PASS_WITH_REMEDIATION**
Blocker yok. F-B4+F-B1 **Phase 6 research dalgaları level verisi tüketmeye başlamadan ÖNCE zorunlu** (F-B4 manifest yanlış veri içeriyor); F-B2/B3/B5 P6-000/P6-001 kapsamında.

```
==================================================
REVIEW COMPLETE
Completed model: FABLE 5 (SUBSTITUTE — Opus 4.8 unavailable)
Next required model: SONNET 5
Review verdict: PASS_WITH_REMEDIATION
Accepted components: candle builder+morphology (tam) · swing extractor (tam) ·
  push geometry (onarım doğrulandı, tam) · duplicate-engine kararı (tam) ·
  level registry (KISMİ — F-B4/F-B1 onarımına tabi)
Blocking findings: YOK (F-B4 Phase-6-öncesi zorunlu ama Phase 4-5 kapanışını geri açmıyor)
Required remediation:
  BATCH-P6-000 (Sonnet, Phase 6 ön-koşul batch'i):
    F-B4: level_registry pencere-kenarı truncation fix + depodaki 4 hatalı
      satırın idempotent yeniden-üretimi + truncation testi
    F-B1: session/day known_at = max(son close, dönem sınırı) + sınır testi
    F-B3: gaps-tablosu çapraz-işaretleme (GAPPED data_quality)
    F-B2: SCHEMA_DICTIONARY uyarısı (touch-stats point-in-time değil)
  BATCH-P6-001 (Sonnet, Phase 6 ilk gerçek batch'i):
    W1 (cycle-integrity dalgası, HISTORICAL_RESEARCH_WAVES.md) hazırlığı +
    F-B5/F3 engine gateway: population assembly TEK modülden
    (assert_not_pooled zorunlu + yalnız ami/chart+warehouse feature'ları +
    evidence_contamination/researcher_exposure_ledger otomatik kaydı) +
    mutation testleri. Prereg-freeze disiplini Phase 6 DoD'una göre.
Required new tests: pencere-kenarı truncation · known_at sınır kilidi ·
  GAPPED işaretleme · gateway mutation testleri (pooled-population,
  ami/chart-dışı feature reddi, ledger-yazımı atlanamaz)
Operator decisions: OD-001/002/004-010 hâlâ OPEN — hiçbiri P6-000/001'i bloklamıyor
Exact resume instruction for Sonnet 5: "SYSTEM_STATE §49'u oku;
  BATCH-P6-000 remediation'ı uygula; testleri koştur; protected diff doğrula;
  sonra BATCH-P6-001 (W1 + gateway) ile Phase 6'ya başla."
==================================================
```

---

## 50. BATCH-P6-000 + BATCH-P6-001 DONE — PHASE 6 ÖN-KOŞULLARI TAMAMLANDI (2026-07-04, Sonnet 5)

**Active phase:** 6-ön-koşul → **KAPANDI**. Gerçek W1 (cycle-integrity) araştırma dalgası HENÜZ BAŞLAMADI — bu batch yalnız FABLE REVIEW B'nin 5 bulgusunu (F-B1..B5) kapattı.

### BATCH-P6-000 (F-B1, F-B3, F-B4, F-B2)

**F-B4 (MANIFEST bug, düzeltildi):** `level_registry.compute_session_levels`/`compute_previous_day_levels` artık her dönemin İLK mumunun gerçek sınırda olduğunu kontrol ediyor (`first_member_ts - boundary < bar_ms`); değilse dönem tamamen atlanıyor. **Gerçek veride teyit edildi:** truncated US session (13:00 origin, yalnız 20:00-21:00 verisiyle) artık listede yok; bağımsız SQL kontrolüyle doğrulandı.

**F-B1 (latent lookahead, düzeltildi):** `known_at_ts = max(son mum close, gerçek dönem-sonu sınırı)`. Gerçek veride 0/8 ihlal (bağımsız doğrulandı).

**F-B3 (düzeltildi):** `candle_builder`, `microstructure.db:gaps` (stream=agg_trades, 20 kayıt) ile çakışan pencereleri artık `data_quality=GAPPED` işaretliyor. Mevcut 48h pencerede örtüşme yok (0 GAPPED) — kod senkron testlerle (sentetik overlap) doğrulandı.

**F-B2 (bloklandı, tam çözülmedi — bilinçli):** `ami_levels.touch_stats_point_in_time` yeni kolon, her satırda **0**. Point-in-time-safe bir yeniden-hesaplama motoru bu batch'in kapsamı DIŞINDA (gelecek iş); bunun yerine gateway seviyesinde erişim bloklandı (F-B5 ile birleşik çözüm).

**Veri onarımı:** Eski `level-v1` satırları (yazılım kusuru, araştırma verdict'i DEĞİL) kontrollü silindi — `_delete_superseded_levels()`, idempotent, deterministik. Schema v4→v5 (M-0007): `touch_stats_point_in_time` kolonu idempotent `ALTER TABLE` ile mevcut DB'ye eklendi (yeni DB'ler CREATE TABLE'da doğrudan alır).

### BATCH-P6-001 (F-B5/F3)

`ami/research/feature_gateway.py` — Phase 6'nın **zorunlu tek erişim noktası**:
- `fetch_events()`: REAL_LIQUIDATION+PROXY_* karışımını `assert_not_pooled` ile reddeder.
- `fetch_level_features()`: F-B2 kolonlarını (`touch_count` vb.) `touch_stats_point_in_time=1` olmadan **reddeder** (şu an her zaman reddedilir, çünkü flag hep 0).
- `fetch_chart_feature()`: yalnız bilinen 7 tablo (ami_events/cycles/candles/morphology/swings/levels/pushes); ami_levels/ami_events kendi özel fonksiyonlarından geçmeli.
- Her başarılı erişim `researcher_exposure_ledger`'a (Phase 2 tablosu, YENİ ledger değil) kaydedilir.

### Testler ve doğrulama
20 yeni/genişletilmiş test (schema +2, candle_builder +3, level_registry +5, feature_gateway +10). **Toplam AMI: 255/255 ✓.** Tüm suite'ler ikişerli gruplar halinde sıralı çalıştırıldı; Phase 1-3 dosyaları da regresyonsuz yeniden doğrulandı. Tüm script'ler 2× koşuldu (idempotent). Protected diff: yalnız pre-existing baseline.

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0007 (schema v5 + level repair) + M-0008 (gateway). `SCHEMA_DICTIONARY.md` Phase 6-öncesi remediation bölümü eklendi.

### Phase 6 gerçek başlangıç durumu
**Hazır — ancak W1 (cycle-integrity dalgası) HENÜZ BAŞLAMADI.** Bu iki batch yalnız ön-koşul temizliğiydi (operatör talimatı kapsamı buydu). Sıradaki gerçek adım: W1 dalgası (HISTORICAL_RESEARCH_WAVES.md) — prereg-freeze + master protokol §9 zorunlu kontrol seti (matched controls, purge/embargo, multiple-testing correction) ile, `feature_gateway` üzerinden. Model değişmedi, Sonnet 5 devam ediyor; W1'e başlamak için operatör onayı gerekmez (research-only) ama operatör "devam et" demeden bir sonraki adıma geçilmeyecek.

---

## 51. BATCH-P6-002 (W1) DONE — CYCLE INTEGRITY & DEDUPLICATION DALGASI TAMAMLANDI (2026-07-04, Sonnet 5)

**Active phase:** 6 (W1) → **KAPANDI**. Descriptive integrity wave — yeni alpha/route iddiası YOK; mevcut bucket'lara dokunulmadı; yeni forward observer başlatılmadı.

### Altyapı
Schema v5→v6 (M-0009): `experiment_registry` + `experiment_results` (Protocol §7.2/§7.7, önceden planlanmış, şimdi kuruldu — "research sonuçları canonical SQL'e" gereksinimini karşılıyor). `feature_gateway`'e `fetch_cycles()` + genel `equals`-filtreli `fetch_chart_feature()` eklendi (event_cycle_membership dahil). `ami/research/w1_cycle_integrity.py` — population TAMAMEN `feature_gateway` üzerinden (doğrudan tablo erişimi yok).

**2 gerçek bug bulunup düzeltildi (kod yazarken):** (1) `_record_exposure`'ın exposure_id'si content-hash+ms-timestamp'ten üretiliyordu — W1'in tek koşumda 8 gateway çağrısı yapması aynı milisaniyede UNIQUE constraint ihlaline yol açtı; `uuid.uuid4()` ile düzeltildi. (2) `experiment_results` INSERT'i idempotent değildi (re-run'da satırlar birikiyordu, 12→24); `DELETE FROM experiment_results WHERE experiment_id=?` eklendi. İkisi de testlerle kilitlendi.

### === W1 KONSOLİDE SONUÇ RAPORU ===

**Deney:** E-W1-CYCLE-INTEGRITY-001 (canonical.sqlite: experiment_registry + experiment_results). software_verdict=PASSED, scientific_verdict=ANSWERED_SUPPORTED.

**1) Test edilen population:** Tüm ETHUSDT `ami_events` (source_quality=REAL_LIQUIDATION, `feature_gateway.fetch_events` ile — REAL/PROXY karışımı `assert_not_pooled` tarafından garanti reddedilir) + karşılık gelen `ami_cycles` (cycle_definition_version=canonical-v1, `fetch_cycles`). Dataset hash deneyde donduruldu.

**2) Raw event N / independent cycle N:**
| Seviye | N | Not |
|---|---|---|
| raw ledger trades | **270** | shadow ledger'daki OPEN satırı (rule_name'li) |
| anchor (ami_events) | **252** | Protocol §8.1 dedup sonrası — "event" |
| **independent cycle (canonical-v1)** | **167** | `ami/identity/cycle_resolver.py`, OD-003 onaylı |

anchor_to_cycle_ratio = **0.66** — yani raw anchor sayısını "bağımsız N" gibi kullanan HERHANGİ bir geçmiş analiz, gerçek bağımsız örneklem sayısını ortalama **~1.5x şişirmiş** olabilir.

**3) Duplicate/overlap etkisi:** 252 anchor'un **16'sı** (event_count>1) birden fazla route'un aynı anchor'a bağlandığı çoklu-route anchor'ları (event_count dağılımı: 236×1 / 15×2 / 1×4). Cycle seviyesinde ayrıca **15/167 cycle (%9)** direction_conflict=1 taşıyor — bunlar Protocol §8.3 gereği WAIT olarak bırakıldı, otomatik yöne ÇÖZÜLMEDİ.

**4) Cooldown sensitivity (non-canonical, is_canonical=0, dokunulmadı):**
| Pencere | 1h | 2h | 4h | 6h | 12h | 24h | **canonical-v1** |
|---|---|---|---|---|---|---|---|
| Episode N | 152 | 111 | 88 | 82 | 55 | 38 | **167** |

Canonical-v1 (167), saf 4h-gap penceresinden (88) daha YÜKSEK — çünkü resolver 4h-gap'e ek olarak yapısal-durum-değişimini de zorla bölüyor (Obs §5.4 çok-sinyalli tasarımın somut etkisi; yalnız gap-tabanlı bir pencereden temelde farklı davranıyor, "4h ile 6h arası" gibi naif bir beklenti YANLIŞ çıktı).

**5) Etkilenen mevcut bucket ve eski sonuçlar (OD-011, YENİ — hiçbir bucket değiştirilmedi):** `data/ami/knowledge.sqlite`'taki 11 Knowledge Object'in (K-S34-HOUR17, K-S34-BOOK-PULL, K-S34-FUNDING-LEVEL, K-S34-MONDAY-VETO, K-S34-MGMT-6H, K-S34-PRECASCADE, K-S34-MECH-COMPOSITE, K-LATENT-REGIME, K-BUYFADE-SILENCE-INFO, vb.) VE SYSTEM_STATE §4'teki route-N tabloları (N=72 TIME_EXIT, N=15 SHORT vb.) **cycle-adjusted independent-N kullanmadan** üretildi. Bu, onların HOLDOUT_VALIDATED/FORWARD_VALIDATING statülerinin yanlış olduğu anlamına gelmez — ama N=0.66x düzeltmesi uygulanmadan bağımsızlık/güven düzeyleri tam doğrulanamaz. **Hiçbir KO veya bucket sessizce değiştirilmedi; OD-011 olarak kayda geçti, operatör kararı bekliyor.**

**6) Sonraki exact research wave:** HISTORICAL_RESEARCH_WAVES.md sırasına göre W2 (Unconditional LONG genesis + failed-fade LONG) bir sonraki adımdır, ANCAK **W2, "all-timestamp candidate universe" (Protocol §17.8) gerektirir — bu henüz YOK (BLOCKED_BY_DATA)**. İki gerçekçi seçenek: **(a)** candidate-universe koleksiyon/hesaplama alt-batch'ini önce inşa et, sonra W2'ye geç; **(b)** P5 chart-engine'leri (candle/swing/level/push) artık HAZIR olduğu için sıralamayı esnetip **W3'e (entry timing, early/T0/delayed/late)** geç — bu, mevcut altyapıyla doğrudan başlatılabilir. **Karar operatöre bırakıldı**, otomatik seçim yapılmadı.

### Testler ve doğrulama
17 yeni/genişletilmiş test (gateway +1 [fetch_cycles], W1 +6, registry_seed OD-011 için 3 test güncellendi). **Toplam AMI: 262/262 ✓.** Tüm script'ler 2× koşuldu (idempotent — experiment_results artık 12/12, önceki 24 satırlık duplikasyon onarıldı). Protected diff: yalnız pre-existing baseline. Hiçbir proses başlatılmadı/durduruldu; hiçbir forward observer aktive edilmedi; hiçbir bucket parametresi değiştirilmedi.

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0009 (schema v6 + 2 bug fix) + M-0010 (W1 veri). `SCHEMA_DICTIONARY.md` "Phase 6 — W1" bölümü. `OPERATOR_DECISION_QUEUE.md` OD-011 (yeni, OPEN). Model değişmedi, Sonnet 5 devam ediyor — operatör onayı olmadan W2/W3'e geçilmeyecek.

## 52. BATCH-P6-003 — ALL-TIMESTAMP CANDIDATE UNIVERSE DONE (2026-07-04, Sonnet 5)

**Active phase:** 6 (W2 hazırlığı) → **KAPANDI**. Operatör talimatı: "W3'e atlama, önce candidate universe'i inşa et." Descriptive/altyapı dalgası — yeni alpha/route iddiası YOK; hiçbir bucket'a dokunulmadı; hiçbir forward observer başlatılmadı.

### Altyapı
Schema v6→v7 (M-0011): `ami_candidate_universe` — her kapanmış mum slotu, event varlığına koşullanmadan bir candidate satırı olur (`is_event_aligned`/`aligned_event_id` yalnız ek bilgi, satır varlığı filtresi değil). `ami/research/candidate_universe.py`: `build_universe()` (saf fonksiyon) + `seed()` (yalnız `feature_gateway.fetch_chart_feature`/`fetch_events` üzerinden — doğrudan tablo erişimi yok) + `freeze_and_record()` (4 denominatörü canonical SQL'e yazar: `E-CANDIDATE-UNIVERSE-001`).

**known-at-safe/gap-aware/deterministic:** `known_at_ts=candle.close_ts_ms` (Obs §6.2 ile aynı disiplin); `data_quality` alttaki mumdan miras (AVAILABLE/GAPPED, hiç sessizce yükseltilmez); `candidate_id=hash(symbol,timeframe,slot_ts_ms,universe_definition_version)` — deterministik, yeniden-üretilebilir.

**KRİTİK BULGU — kapsam uyuşmazlığı (kod yazılırken bulundu, gizlenmedi):** `ami_candles` Phase 4'ten kalma bilinçli `lookback_hours=48` sınırlı pencereyle inşa edilmiş (`ami/chart/candle_builder.py`). Candidate universe bu dar ~48h pencereyi miras alıyor, ANCAK `event_n`/`anchor_n`/`independent_cycle_n` (W1) TÜM tarihi kapsıyor (2026-02-17 → bugün). Bu ikisini doğrudan oranlamak yanıltıcı olurdu — bu yüzden `freeze_and_record()` HEM all-history W1 sayılarını HEM DE pencere-kapsamlı `anchor_n_in_candidate_window`/`cycle_n_in_candidate_window` sayılarını AYRI metrik olarak yazıyor; `candidate_to_anchor_ratio`/`candidate_to_cycle_ratio` yalnız pencere-kapsamlı sayıyı kullanıyor (all-history sayıyı asla payda/pay olarak karıştırmıyor).

### 4 denominatör (gerçek veri, ETHUSDT, canonical.sqlite)
| Seviye | N | Kapsam |
|---|---|---|
| **raw_candidate_n** | **2932** | candle penceresi: 2026-07-01T20:00 → 2026-07-03T20:51 UTC (48h, Phase 4 lookback) |
| — event-aligned | 22 | pencere içi |
| — no-event | 2910 | pencere içi |
| anchor_n (pencere-kapsamlı) | 22 | aynı 48h pencere |
| cycle_n (pencere-kapsamlı) | 17 | aynı 48h pencere |
| event_n (all-history) | 270 | W1, tüm tarih |
| anchor_n (all-history) | 252 | W1, tüm tarih |
| independent_cycle_n (all-history) | 167 | W1, tüm tarih |

candidate_to_anchor_ratio=0.007503, candidate_to_cycle_ratio=0.005798 (yalnız pencere-kapsamlı sayılarla, doğru referans).

### OD-011 zenginleştirme (operatör talimatı: per-KO AFFECTED/RECOMPUTE_REQUIRED)
`CONTRADICTION_REGISTER.md`'ye **CT-005** eklendi: 11 KO'nun (`data/ami/knowledge.sqlite`) 8'i **RECOMPUTE_REQUIRED** (K-S34-HOUR17-001 [FORWARD_VALIDATING, deploy adayı, en yüksek öncelik], K-S34-BOOK-PULL-001, K-S34-FUNDING-LEVEL-001, K-S34-MECH-COMPOSITE-001, K-S34-MGMT-6H-001, K-S34-MONDAY-VETO-001, K-S34-PRECASCADE-001, K-S34-SCALEIN-100-001), 3'ü **AFFECTED** (descriptive, N-bağımlı istatistiksel iddia taşımıyor: K-BUYFADE-SILENCE-INFO-001, K-LATENT-REGIME-001, K-S34-REFILL-CTX-001). `knowledge.sqlite` DEĞİŞTİRİLMEDİ (governance-layer riski — yalnız flag). OD-011 metni bu sınıflamaya referans verecek şekilde güncellendi; **canonical cycle-adjusted recomputation ayrı, operatör onaylı bir Phase 6 dalgası olarak planlanacak** — bu batch onu YÜRÜTMEDİ.

### Testler ve doğrulama
16 yeni test (candidate_universe modülü +12 [8 build/seed + 4 freeze_and_record/scope], registry_seed CT-005 için 3 test güncellendi → contradiction_registry 14→15). **Toplam AMI: 274/274 ✓** (262+12). Gerçek DB'de 2× çalıştırıldı (`ami_candidate_universe`: 2932=2932 satır, candidate_id 2932 distinct — idempotent, SQL ile bağımsız doğrulandı; `experiment_results` E-CANDIDATE-UNIVERSE-001: 12/12, W1'in kendi deneyi DOKUNULMADI). Protected diff: yalnız pre-existing baseline (`tools/s34_state_machine_live_executor.py` untracked, değişmedi).

### Sonraki adım
Operatör sıralaması: candidate universe tamamlanıp testleri geçti → **W2 (Unconditional LONG genesis + failed-fade LONG)** çalıştırılabilir, ardından **W3 (entry timing)**. W2/W3 başlamadan önce ayrı onay istenecek.

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0011 (schema v7 + candidate universe verisi). `SCHEMA_DICTIONARY.md` "Phase 6 — Candidate Universe" bölümü. `OPERATOR_DECISION_QUEUE.md` OD-011 zenginleştirildi (statü hâlâ OPEN). `CONTRADICTION_REGISTER.md` CT-005 (yeni, AÇIK-KAYITLI). Model değişmedi, Sonnet 5 devam ediyor.

## 53. BATCH-P6-003b — CANDLE FULL-HISTORY BACKFILL + W2 MEZARLIK ÇAKIŞMASI → W2 ERTELENDİ (2026-07-04, Sonnet 5)

**Active phase:** 6 (W2→W3 geçişi) → W2 **ERTELENDİ** (mezarlık çakışması), altyapı TAMAMLANDI, W3'e geçiliyor. Operatör talimatı: "W2 LONG genesis'i çalıştır" idi; kod yazılırken/veri kontrol edilirken W2'nin mezarlıkla çakıştığı bulundu — operatöre soruldu, operatör "OI retry-koşulunu kontrol et" (seçenek 2) dedi; koşul zayıf çıktı, operatör "W2'yi bırak, W3'e geç" (seçenek 1) ile kapattı.

### Candle full-history backfill (altyapı, W2'den bağımsız kalıcı değer — W3/W4/vb. için de gerekli)
`ami_candles` Phase 4'ten kalma `lookback_hours=48` sınırıyla inşa edilmişti (yalnız 22/252 anchor kapsıyordu — bkz. §52 kapsam uyarısı). Tam-tarih ihtiyacı için **yeni, verimli streaming builder** yazıldı: `ami/chart/candle_builder.py:build_candles_streaming()` — `agg_trades`'i (ETHUSDT, 173M satır) TEK ORDER BY ts_ms geçişte (`fetchmany` chunk'larıyla, bellek sınırlı) tarar; eski per-bucket implementasyon (~200K ayrı sorgu, saatler sürerdi + potansiyel 40GB+ RAM) yerine. `derive_higher_timeframe()` — 5m'i 1m sonuçlarından Python'da türetir (ikinci pahalı DB taraması YOK). `seed_full_history()` — ikisini birleştirir, idempotent upsert.

**Gerçek bug bulunup düzeltildi (kod yazılırken, cross-check testiyle yakalandı):** streaming builder'ın SQL alt-sınırı `start_ts_ms` kullanıyordu ama bucket hizalaması `bucket_open<=start_ts_ms` olabiliyordu — ilk bucket'ın `[bucket_open, start_ts_ms)` kısmı sessizce kayboluyordu (gerçek veride 3h pencerede trade_count 499→184 farkı ile yakalandı). Düzeltme: SQL alt-sınırı da `bucket_open`'a hizalandı. Düzeltme sonrası 3h VE 24h pencerede referans (`build_candles`) ile **tam eşitlik** doğrulandı (5m'de yalnız kayan-nokta toplama-sırası gürültüsü, max fark 2.9e-11 — gerçek uyuşmazlık değil).

**Gerçek veri sonucu:** 173,464×1m + 34,801×5m candle, **910.8 saniyede** (15.2 dk, tek sıralı proses, paralel yok). 0 OHLC ihlali, distinct=total (2932→173464 arası hiç çakışma yok). data_quality: 173,436 AVAILABLE + 28 GAPPED (gaps tablosuyla dürüstçe kesişim). **252/252 anchor artık candle penceresi içinde** (önceki §52'deki 22/252 kapsam sorunu ÇÖZÜLDÜ).

### W2 mezarlık çakışması (kritik bulgu, gizlenmedi)
`data/ami/knowledge.sqlite:failure_archive` kontrol edildiğinde, `HISTORICAL_RESEARCH_WAVES.md`'deki W2 tanımı ("Unconditional LONG genesis + failed-fade LONG") **iki graveyard kaydıyla doğrudan çakışıyor**:
- **#8** "BUY-side short-squeeze fade (all variants)" — NO_EDGE, "5:159 positive ratio", retry_condition **BOŞ** (kalıcı kapalı).
- **#17** "BUYFADE pre-event LONG genesis/maturity T0 filtresi" — NO_EDGE, retry_condition: "OI verisi birikince leverage-genesis boyutuyla yeni prereg".

CLAUDE.md kesin guardrail'ı ("buy-side fade... tekrar test etme") ve Protocol §5 mezarlık disiplini gereği, W2 bu haliyle ÇALIŞTIRILMADI.

**#17 retry-koşulu kontrol edildi, ZAYIF çıktı:** `microstructure.db:open_interest` ETHUSDT — nominal aralık 2026-03-28→2026-07-03 (2370 satır) ama **tek bir 1807 saatlik (75 gün) boşluk** var (2026-04-18→2026-07-02). Gerçek kullanılabilir kapsama yalnız 2 kopuk pencere: (a) 2026-03-28→04-18, saatlik, 27 anchor; (b) 2026-07-02→07-03, 60sn, 11 anchor. **Toplam gerçek-OI-kapsamlı anchor: 38/252 (%15)**, iki farklı çözünürlükte ve temporal olarak KOPUK (train/test ayrımı aslında rejim-ayrımına dönüşür). Bu N ile TRAIN/TEST+MC zorlamak neredeyse kesin yeni bir INSUFFICIENT_SAMPLE mezar kaydı üretir.

**Operatör kararı: W2 ERTELENDİ** (retry-koşulu "gerçek anlamda karşılanmadı" olarak kayda geçti — OD-012, bkz. aşağı), **W3'e (entry timing) geçiliyor**. Mezarlık ailesi (buy-side fade/failed-fade LONG) tekrar açılmadı; hiçbir yeni prereg/deney kaydı YAZILMADI (yalnız kontrol yapıldı, sonuç negatif).

### Testler ve doğrulama
8 yeni test (`test_ami_chart_candle_builder.py`: streaming×4, derive_higher_timeframe×3, seed_full_history idempotency×1 — streaming/derive gerçek veriyle 3h+24h cross-check, ayrı script ile, testin kendisi synthetic fixture kullanıyor). **Toplam AMI: 282/282 ✓** (274+8). Gerçek DB'de idempotency UNIQUE constraint ile yapısal garanti + testlerle kilitli (tam 15dk'lık real-data re-run tekrarlanmadı — maliyet/fayda, UPSERT zaten test edildi). Protected diff: yalnız pre-existing baseline.

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0012 (candle full-history backfill verisi). `OPERATOR_DECISION_QUEUE.md` OD-012 (yeni: W2 ertelendi, retry-koşulu zayıf). `HISTORICAL_RESEARCH_WAVES.md`'ye not: W2 önkoşulu artık "candidate universe" DEĞİL "OI verisinin gerçek/sürekli birikmesi" (mevcut kopuk/düşük-yoğunluk kapsamı yetersiz). Model değişmedi, Sonnet 5 devam ediyor — **sıradaki: W3 (entry timing)**.

## 54. BATCH-P6-004 — W3 ENTRY-TIMING DUE-DILIGENCE RECONCILIATION: W3 ZORLA AÇILMADI, W4'E GEÇİLDİ (2026-07-04, Sonnet 5)

**Active phase:** 6 (W3 due-diligence) → W3 **ZORLA AÇILMADI** (§53'teki W2 mezarlık dersinden sonra aynı disiplin W3'e de uygulandı). Operatör talimatı: entry-timing alanındaki mevcut 10 raporu + `failure_archive`'ı read-only reconcile et; verdict taksonomisi (EXACT_HYPOTHESIS_ALREADY_TESTED/SCIENTIFICALLY_REJECTED/ECONOMICALLY_REJECTED/RETRY_CONDITION_NOT_MET/RETRY_CONDITION_MET/DISTINCT_MECHANISM/UNANSWERED_RESEARCH_GAP/SUPERSEDED/INSUFFICIENT_SAMPLE) kullan; gerçek yeni soru yoksa W3'ü zorlama, sentezi kaydet, sıradaki bağımsız dalgaya geç.

### Reconciliation sonucu (10 rapor, `ami/research/w3_entry_timing_reconciliation.py`, canonical SQL'e yazıldı — E-W3-ENTRY-TIMING-RECONCILIATION-001)
| Rapor | Verdict | Eşleşen failure_archive |
|---|---|---|
| S34_ABSORPTION_TIMING.md (N=36) | UNANSWERED_RESEARCH_GAP | — |
| S34_ABSORPTION_TIMING_BROAD.md (N=51) | UNANSWERED_RESEARCH_GAP | — |
| S34_CONFIRMATION_ENTRY.md | EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED | #8, #16 |
| S34_EARLY_BUILD_ENTRY.md | EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED;RETRY_CONDITION_NOT_MET | #1 |
| S34_ENTRY_OFFSET_DECOMPOSITION.md | SUPERSEDED (execution-realism diagnostic, kendi etiketi "not a new rule") | — |
| S34_ENTRY_OFFSET_DECOMPOSITION_50K_TP120.md | SUPERSEDED | — |
| S34_MICRO_ENTRY_SCALP.md | EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED | #2 |
| S34_SELL_DELAYED_LONG_SCAN.md | EXACT_HYPOTHESIS_ALREADY_TESTED;ECONOMICALLY_REJECTED;RETRY_CONDITION_NOT_MET | #16 (DELAY600, #16'nın kendi reason'ında geçiyor) |
| S34_V_ENGINE_CONFIRMATION_DELAY_SWEEP.md (N=22) | EXACT_HYPOTHESIS_ALREADY_TESTED;INSUFFICIENT_SAMPLE | #2, #16 |
| S34_V02_ENTRY_QUALITY_NAVIGATION.md (N=11) | INSUFFICIENT_SAMPLE (operasyonel dashboard etiketleme, hipotez testi değil) | — |

**Sonuç:** 9/10 rapor ya graveyarded bir entry-timing mekanizmasını (delay/confirm/pullback/wait-for-reclaim) aynı negatif ekonomik verdict'le tekrar test ediyor, ya known-at disipliniyle zaten süperseded bir execution-diagnostic, ya da N çok küçük operasyonel bir dashboard etiketlemesi. **TEK gerçek açık soru** (absorption-timing point-in-time permission-feature geçerliliği, N=36-51) gerçek ve mezarlıktan FARKLI ama §53'teki OI sorunuyla aynı büyüklükte yetersiz-örneklem riski taşıyor. **Operatör talimatı gereği W3 bu temelde ZORLANMADI**; hiçbir yeni prereg/deney açılmadı, hiçbir mezarlık ailesi tekrar test edilmedi.

### Testler ve doğrulama
6 yeni test (`test_ami_research_w3_entry_timing_reconciliation.py`: taksonomi uyumu, RETRY_CONDITION_MET olmadığı kilidi, W3-zorlanmadı kilidi, canonical-SQL yazım + idempotency). **Toplam AMI: 288/288 ✓** (282+6). Protected diff: yalnız pre-existing baseline. Hiçbir bucket/route/forward observer değişmedi.

### Kalıcı kayıtlar
`HISTORICAL_RESEARCH_WAVES.md`'ye W3 notu eklendi (ZORLA AÇILMADI, reconciliation referansı). Yeni MD oluşturulmadı (talimat gereği); sentez yalnız canonical SQL + SYSTEM_STATE'e yazıldı. Model değişmedi, Sonnet 5 devam ediyor — **sıradaki aday: W4 (post-event path taxonomy + structural location + event geometry, önkoşul P3 paths — zaten sağlanmış)**, operatör onayı bekleniyor.

## 55. BATCH-P6-005 — W4 POST-EVENT PATH TAXONOMY PREREGISTER + ÇALIŞTIRILDI (2026-07-04, Sonnet 5)

**Active phase:** 6 (W4) → **TAMAMLANDI**. Descriptive/taxonomy dalgası — ekonomik iddia YOK, mezarlıktaki "reversal fade" hipotezinin (S34_REVERSAL_BACKTEST/S34_REVERSAL_STOP_BACKTEST, ETH N=346 stop-sweep 0 leads) yeniden testi DEĞİL. Hiçbir bucket/route/observer değişmedi.

### Altyapı ön-hazırlığı (M-0013): yapısal motorlar tam-tarihe genişletildi
`ami_swings`/`ami_levels`/`ami_pushes` zaten lookback sınırı taşımıyordu — `ami_candles` büyüyünce (§53) kod değişikliği olmadan otomatik genişlediler: **32,055 swing** (0.76s), **33,259 level** (388.4s — touch-stat O(level×candle), F-B2 ile araştırmaya zaten bloklu), **24,940 push** (179.9s); 0 ihlal. `ami_candidate_universe` yeniden çalıştırıldı: **173,464 candidate, 252/252 anchor, cycle_n_in_window=167=167 (all-history ile birebir)** — §52'nin 48h-pencere kapsam sorunu tamamen çözüldü. `feature_gateway`'e `ami_candidate_universe` eklendi (yeni negative-control kaynağı).

### W4 preregistration (dondurulmuş, sonuca bakmadan önce — `ami/research/w4_post_event_path_taxonomy.py` docstring'i = freeze)
- **Popülasyon:** ETHUSDT REAL_LIQUIDATION, anchor_n=252, independent_cycle_n=167 (canonical-v1, W1 ile AYNI popülasyon — OD-011 dersi gereği raw-N değil cycle-N esas denominatör).
- **Ufuklar (sabit):** scalp_30m, scalp_1h, swing_4h, swing_24h.
- **Outcome taksonomisi (sabit, fit edilmedi):** CONTINUATION (≤-20bps)/REVERSAL (≥+20bps)/CHOP (arası); referans fiyat = anchor'dan önceki son kapanmış 1m mum (known-at-safe).
- **Yapısal-konum bayrakları (sabit, known-at-safe):** near_swing_low (≤50bps), near_level (≤50bps), recent_down_push (son 4h içinde biten DOWN push).
- **Multiple-testing ailesi (5 sabit karşılaştırma, FAM_POST_EVENT_PATH_TAXONOMY):** C1 scalp-vs-swing, C2 near_swing_low, C3 near_level, C4 recent_down_push, C5 anchor-vs-rastgele-zaman (candidate universe).
- **Kontroller:** kronolojik %70/%30 stabilite kontrolü (eşik fit etmek için değil — fit edilecek eşik yok); candidate-universe rastgele-zaman negative control.
- **Stop condition:** bucket N<20 → insufficient_sample=True (sessizce atlanmaz).

### Gerçek sonuçlar (E-W4-POST-EVENT-PATH-TAXONOMY-001, canonical.sqlite)
| Karşılaştırma | Sonuç |
|---|---|
| **C1 scalp vs swing** | scalp (N=504): CONTINUATION 38.7%/REVERSAL 33.1%/CHOP 28.2%. swing (N=494): CONTINUATION 36.8%/**REVERSAL 54.3%**/CHOP 8.9%. Ufuk uzadıkça REVERSAL payı belirgin artıyor, CHOP daralıyor. |
| **C2/C3/C4 (yapısal-konum)** | **DEGENERATE:** "no" bucket'ı üç bayrak için de N=0 (insufficient_sample=True). Neden: swing_low/level/push bu piyasada çok yoğun (sırasıyla ~12.5dk/~6dk/~16dk'da bir) — sabit 50bps/4h eşiği ayırt edici değil, HER anchor "yakın" çıkıyor. **Eşik geriye dönük DEĞİŞTİRİLMEDİ** (post-hoc fishing olurdu); dürüstçe DEGENERATE/INSUFFICIENT_SAMPLE raporlandı. |
| **C5 anchor vs rastgele-zaman** | Anchor popülasyonu (swing_24h, N=243): REVERSAL 57.6%/CONTINUATION 36.6%/CHOP 5.8%. Rastgele-zaman kontrolü (N=251): REVERSAL 47.0%/CONTINUATION 45.4%/CHOP 7.6%. **Anchor'lar rastgeleden ~10.6 puan daha REVERSAL-ağırlıklı** — descriptive mean-reversion-after-cascade etkisi GERÇEK (önceki "reversal base rate REAL ama harvestable değil" bulgusuyla bağımsız yöntemle tutarlı doğrulama). |
| **Stabilite (train/test, kronolojik)** | TRAIN (N=176): REVERSAL 54.5%. TEST (N=67): REVERSAL 65.7%. Yön tutarlı (REVERSAL her iki dilimde de baskın), büyüklük küçük-N gürültüsüyle dalgalanıyor. |

### LONG/SHORT etkisi ve scalp/swing ayrımı (operatör sorusu)
Ledger TEK yönlü (event_family="...SELL...", tüm 252 anchor SELL-cascade) — LONG-anchor/SHORT-anchor karşılaştırması YOK (popülasyonda LONG anchor bulunmuyor, bu açıkça not edildi, uydurulmadı). Bunun yerine SELL-anchor'ların LONG-yönlü (REVERSAL) mü SHORT-yönlü (CONTINUATION) mü sonuçlandığı ölçüldü: **swing ufkunda LONG-yönlü (REVERSAL) baskın (54.3%), scalp ufkunda daha dengeli (33.1% REVERSAL/38.7% CONTINUATION)** — yani cascade sonrası ilk 30-60dk'da yön belirsiz, 4-24h'de LONG-yönlü toparlanma descriptive olarak daha olası. Bu, **tradeable bir iddia DEĞİL** (mezarlıktaki reversal-fade zaten ekonomik olarak reddedildi) — yalnız piyasa-tanımı düzeyinde bir gözlem.

### Testler ve doğrulama
11 yeni test (`test_ami_research_w4_post_event_path_taxonomy.py` 10/10: classify_path sınırları, known-at-safe candle index, structural-flag point-in-time koruması, sentetik+gerçek-veri end-to-end, canonical-SQL yazım+idempotency; `test_ami_research_feature_gateway.py` +1: ami_candidate_universe erişimi). **Toplam AMI: 299/299 ✓** (288+11). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=8, idempotent).

### Sonraki exact research wave (operatör kararına bırakıldı)
İki aday: **(a)** C2-C4'ün degenerate çıkmasını çözecek bir takip dalgası — daha sıkı/percentile-tabanlı yapısal-konum eşikleri (örn. 10bps veya üst-timeframe swing/level) ile YENİ bir prereg (bu dalganın post-hoc keşfi, ayrı hipotez olarak açılmalı — Protocol §2). **(b)** `HISTORICAL_RESEARCH_WAVES.md`'deki sıradaki dalga: **W5** (candle morphology + swing grammar + sweep + breakout/retest + unconditional SHORT genesis, önkoşul P4-5 + Opus B — SHORT tarafı, LONG'un W2'de ertelenmiş olmasına paralel bir mekanik).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0013 (yapısal motor tam-tarih refresh) + M-0014 (W4 prereg+veri). Yeni MD oluşturulmadı. Model değişmedi, Sonnet 5 devam ediyor — sıradaki dalga için operatör onayı bekleniyor.

## 56. BATCH-P6-006 — W5a (CANDLE MORPHOLOGY + SWING GRAMMAR) ÇALIŞTIRILDI; SWEEP/BREAKOUT-RETEST + SHORT-GENESIS BACKLOG'A ALINDI (2026-07-04, Sonnet 5)

**Active phase:** 6 (W5) → W5'in **2/4 bileşeni TAMAMLANDI** (candle morphology + swing grammar), **2/4 bileşeni BACKLOG'A ALINDI** (sweep/breakout-retest, unconditional SHORT genesis — OD-014). Operatör talimatı: W5'in tamamını çalıştır, C2-C4 eşiğini optimize etme (backlog'a kaydet — OD-013 zaten yapıldı), bucket/route/observer değiştirme.

### OD-013 backlog kaydı
W4'ün C2-C4 degenerate bulgusu `OPERATOR_DECISION_QUEUE.md`'ye OD-013 olarak kaydedildi (bu dalgada eşik değiştirilmedi).

### Ön-hazırlık: candle morphology tam-tarihe genişletildi
`ami_candle_morphology.seed()` zaten lookback sınırı taşımıyordu — kod değişikliği olmadan yeniden çalıştırıldı: **208,265 satır** (173,464×1m+34,801×5m), 6.6s, idempotent doğrulandı.

### W5 due-diligence (mezarlık + altyapı kontrolü, W2/W3 dersinden sonra standart hale geldi)
`failure_archive`'da "SHORT genesis"/"sweep"/"breakout"/"morphology"/"grammar" için doğrudan çakışma bulunamadı. Ancak: **(1)** `reports/research/s34/S34_SHORT_CONVICTION.md`/`S34_SHORT_SCORE3_GAUNTLET.md` gibi mevcut raporlar zaten VALİDE, AKTİF bir SHORT mekanizması gösteriyor (BTC-liq-anchored, WR=69-100%, mc_p≤0.032, wf 4-5/5) — ama bu bizim ETHUSDT-SELL-cascade popülasyonumuzdan TAMAMEN FARKLI bir mekanizma, çakışma yok, sadece not edildi. **(2)** "Sweep/breakout-retest" chart-native nesnesi henüz NOT_IMPLEMENTED (SCHEMA_DICTIONARY: volume-profile/breakout altyapısı yok). **(3)** "Unconditional SHORT genesis" whitepaper §29 SHORT ailelerini (buyer exhaustion, failed breakout, OI expansion, funding crowding, cross-exchange zayıflık) gerektiriyor — bunlar henüz `ami/chart/*` feature'ı olarak hesaplanmıyor (W2'yi bloke eden AYNI altyapı-eksikliği sınıfı). **Bu 2 bileşen zorlanmadı, OD-014 ile backlog'a kaydedildi.**

### W5a preregistration + sonuçlar (E-W5A-MORPHOLOGY-SWING-GRAMMAR-001)
`ami/research/w5a_morphology_swing_grammar.py` — **Popülasyon:** W1/W4 ile AYNI (anchor_n=252). **Outcome:** W4'ten AYNEN yeniden kullanıldı (swing_24h path class, yeniden tanımlanmadı — target-shopping'den kaçınıldı). **2 yeni sabit özellik:** anchor candle morphology label (CLOSE_NEAR_HIGH/CLOSE_NEAR_LOW/MID_RANGE_CLOSE), pre-anchor swing-grammar (Dow-theory HH/HL vs LH/LL, known-at-safe). **2 sabit karşılaştırma** (FAM_CANDLE_MORPHOLOGY_SWING_GRAMMAR).

**Gerçek bulgu (null-sonuç, geçerli ve raporlandı):**
| Özellik | Bucket'lar | REVERSAL oranı |
|---|---|---|
| Candle morphology | CLOSE_NEAR_HIGH(n=24)/MID_RANGE(n=58)/CLOSE_NEAR_LOW(n=161) | 58.3% / 60.3% / 56.5% — **ayırt edici değil** |
| Swing grammar | UPTREND(n=54)/DOWNTREND(n=94)/MIXED(n=95) | 59.3% / 57.4% / 56.8% — **ayırt edici değil** |

Her iki özellik de W4'ün genel REVERSAL baz oranını (~%57) anlamlı biçimde koşullandırmıyor — tüm bucket'lar birbirine çok yakın. Bu geçerli bir bulgu: gelecekteki dalgalar için arama alanını daraltıyor (morphology/grammar tek başına ayırt edici değil, ama bu onları "yanlış" yapmaz — yalnızca bu iki koşullandırma AŞAMASINDA anlamlı sinyal yok).

### Testler ve doğrulama
8 yeni test (`test_ami_research_w5a_morphology_swing_grammar.py`: swing-grammar sınıflandırma senaryoları [uptrend/downtrend/mixed/insufficient], known-at-safe koruma, sentetik+gerçek-veri end-to-end, canonical-SQL yazım+idempotency). **Toplam AMI: 307/307 ✓** (299+8). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=4, idempotent).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0015 (candle morphology tam-tarih refresh + W5a prereg/veri). `OPERATOR_DECISION_QUEUE.md` OD-014 (yeni, sweep/breakout-retest+SHORT-genesis backlog). `HISTORICAL_RESEARCH_WAVES.md`'ye W5 notu eklendi. Yeni MD oluşturulmadı. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım için operatör onayı bekleniyor (W5b altyapı-inşa dalgası, veya C2-C4 takip dalgası, veya başka bir Phase 6/7 dalgası).

## 57. BATCH-P6-007 — W6 (COMPRESSION + RELATIVE STRENGTH + SESSION) ÇALIŞTIRILDI; CHANNEL BACKLOG'A ALINDI (2026-07-04, Sonnet 5)

**Active phase:** 6 (W6) → W6'nın **3/4 bileşeni TAMAMLANDI** (compression, relative strength, session), **1/4 bileşeni BACKLOG'A ALINDI** (channel — OD-015, CHANNEL_BOUNDARY NOT_IMPLEMENTED). Operatör talimatı: her bileşeni ayrı preregister et, coverage'ı önce kontrol et, W4'ün population/outcome tanımını koru, W5 null sonucunu değiştirme, bucket/route/observer'a dokunma.

### Coverage kontrolü (başlamadan önce, operatör talimatı gereği)
| Bileşen | Durum | Kanıt |
|---|---|---|
| Compression | **HAZIR** | `ami/states/engine.py:StructurePhase.COMPRESSION` — Phase 3'te `cycle_resolver.py` tarafından zaten kullanılıyor (1h TF, rule-based vol-compression) |
| Relative Strength | **BUILDABLE** | `microstructure.db:mark_prices` BTC+ETH ikisi de tam/sürekli kapsıyor (8.59M satır, 2026-02-17→bugün) — hesaplama yoktu, veri boşluğu yoktu, ilk kez hesaplandı |
| Session | **HAZIR** | `ami/chart/level_registry.py`'deki ASIA/EUROPE/US/OFF sınırları doğrudan yeniden kullanıldı |
| Channel | **NOT_IMPLEMENTED** | `CHANNEL_BOUNDARY` level_type'ı hiç inşa edilmedi (trendline/channel-fitting motoru yok) — **OD-015 ile backlog'a alındı, zorlanmadı** |

### W6 preregistration (E-W6-COMPRESSION-RS-SESSION-001)
`ami/research/w6_compression_rs_session.py` — **Popülasyon ve outcome W4'ten AYNEN korundu** (anchor_n=252, swing_24h path class, yeniden tanımlanmadı). **3 yeni sabit özellik:** compression_at_anchor (COMPRESSION/NOT_COMPRESSION, StructurePhase üzerinden), rs_at_anchor (ETH 1h getirisi − BTC 1h getirisi işareti, 0'da sabit ayrım, fit edilmedi), session_at_anchor (ASIA/EUROPE/US/OFF). **3 sabit karşılaştırma** (FAM_COMPRESSION_RS_SESSION). Kronolojik 70/30 stabilite kontrolü W4/W5a ile aynı.

### Gerçek sonuçlar
| Karşılaştırma | Sonuç |
|---|---|
| **E1 Compression** | NOT_COMPRESSION (n=240, %95.2): REVERSAL 58.3%. COMPRESSION (n=3, %1.2): **insufficient_sample**. Cascade anları ile compression yapısal olarak neredeyse hiç örtüşmüyor — beklenen, mantıklı bir degenerasyon (bug değil). |
| **E2 Relative Strength** | **Anlamlı fark:** RS_ETH_STRONG (n=69): REVERSAL **%68.1**. RS_ETH_WEAK (n=174): REVERSAL **%53.4**. ~15 puan fark — ETH, BTC'ye göre cascade sırasında bile görece güçlüyse (satış idiosinkratik/tükenmiş), toparlanma belirgin daha olası. |
| **E3 Session** | ASIA (n=57): REVERSAL 56.1%. US (n=163): REVERSAL 56.4% — **fark yok**. EUROPE (n=5)/OFF (n=18): insufficient_sample. |

W5'in null sonucu (morphology/swing-grammar ayırt edici değil) bu dalgada DEĞİŞTİRİLMEDİ/kurtarılmaya çalışılmadı — yalnız yeni, bağımsız 3 özellik test edildi.

### Testler ve doğrulama
7 yeni test (`test_ami_research_w6_compression_rs_session.py`: session sınır testleri, RS işaret mantığı [güçlü/zayıf/veri-yok], compression binary etiket, sentetik+gerçek-veri end-to-end, canonical-SQL yazım+idempotency — StateEngine synthetic microstructure.db fixture'ıyla test edildi). **Toplam AMI: 314/314 ✓** (307+7). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=5, idempotent).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0016 (W6 prereg+veri). `OPERATOR_DECISION_QUEUE.md` OD-015 (yeni, channel backlog). `HISTORICAL_RESEARCH_WAVES.md`'ye W6 notu eklendi. Yeni MD oluşturulmadı. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım için operatör onayı bekleniyor.

## 58. BATCH-P6-008 — W6-RS CONFIRMATION WAVE: NÜANSLI SONUÇ, EKONOMİK İDDİA/PROMOSYON YOK (2026-07-04, Sonnet 5)

**Active phase:** 6 (W6-RS confirmation) → **TAMAMLANDI**. Tek, dondurulmuş hipotez sınandı — RS_threshold=0/RS_lookback=1h AYNEN korundu, hiçbir sweep yapılmadı. Popülasyon+outcome W4/W6'dan aynen. Mezarlıktaki reversal-fade hipotezi tekrar taranmadı; hiçbir trade/PnL iddiası veya bucket promotion yapılmadı.

**Exact hypothesis:** SELL-cascade anchor'ında known-at-safe ETH_1h_return − BTC_1h_return > 0 olduğunda, 24h REVERSAL olasılığı RS≤0 grubundan yüksektir.

### 1) Feature coverage + eksik anchor nedeni
total_anchor_n=252, analyzed_n=**243**, excluded_n=9 — **9'unun 9'u da** (`excluded_no_horizon_data`) aynı, tek nedenle açıklanıyor: `anchor_ts+24h`, mevcut candle verisinin kapsadığı son noktayı (`max_candle_close_ts`) aşıyor — bu en yakın zamanlı anchor'lar için 24h ufku henüz "gerçekleşmedi" (veri-güncelliği sınırı, tasarım/kod hatası değil). Sessizce atlanan yok.

### 2) Independent-cycle N
**159** (243 analiz edilen anchor'ın eşleştiği farklı canonical-v1 cycle sayısı — W1'in tüm-tarih 167'sinden biraz düşük, çünkü bu 243'lük alt-küme daha dar).

### 3) Kronolojik 70/30 + AYRI AYRI RS-koşullu replikasyon (asıl confirmatory kontrol)
| Split | N | RS_STRONG reversal | RS_WEAK reversal | Fark |
|---|---|---|---|---|
| TRAIN | 170 | 67.5% (n=40) | 50.8% (n=130) | +16.7pp |
| TEST | 73 | 69.0% (n=29) | 61.4% (n=44) | +7.6pp |

Yön HER İKİ dilimde de tutarlı (RS_STRONG>RS_WEAK), ama etki TEST'te belirgin küçülüyor (+16.7→+7.6pp) — zayıflayan ama yön-değiştirmeyen bir replikasyon.

### 4) Aylık/rolling stabilite
2026-02(+30.2pp)/03(+21.2pp)/04(+27.9pp) yönde tutarlı ama n küçük (çoğu insufficient_sample). **2026-06'da yön TERSİNE döndü** (RS_STRONG %46.2 < RS_WEAK %56.6, −10.4pp) — n=26/76, yeterli örneklemle gerçek bir tutarsızlık. 2026-07 n çok küçük (10/3), bilgi verici değil. **2026-05: 0 anchor** — o ay büyük SELL-cascade oluşmamış (doğrulandı, gerçek boşluk, veri hatası değil).

### 5) Effect size + uncertainty
Risk difference (tüm popülasyon) = **+14.67pp** (RS_STRONG 68.1% [Wilson95: 56.4-77.9%] vs RS_WEAK 53.5% [46.0-60.7%]) — **CI'lar kısmen örtüşüyor**. One-sided label-permutation testi (n=2000, sabit seed): **p=0.0255** — geleneksel 0.05 eşiğinin altında ama güçlü değil, sınırda.

### 6) Candidate-universe negative control (KRİTİK BULGU)
Rastgele-zaman kontrolünde de BENZER büyüklükte bir fark var: RS_STRONG %52.8 vs RS_WEAK %41.0 (**+11.8pp**, n=125/117). **Bu, RS→REVERSAL ilişkisinin cascade'e ÖZGÜ olmadığını, genel bir piyasa-rejimi etkisi olabileceğini gösteriyor** — anchor popülasyonundaki +14.67pp'nin çoğu, cascade sonrası spesifik bir mekanizmadan değil, RS'nin genel olarak sonraki fiyat davranışıyla ilişkili olmasından kaynaklanıyor olabilir.

### 7) Preregistered confounder kontrolleri
- **notional (event size):** RS_STRONG ort=372,458 vs RS_WEAK ort=296,285 — RS_STRONG anchor'lar ortalama ~%26 daha büyük.
- **day_trend_bps:** RS_STRONG ort=−77.2 vs RS_WEAK ort=−141.6 — **belirgin fark**, RS_STRONG anchor'lar daha az ayı-eğilimli günlerde oluşuyor. Plausible confounder (RS ve day_trend muhtemelen korele).
- **session:** RS_STRONG %74 US-ağırlıklı, RS_WEAK %64 US + %27 ASIA — hafif farklı dağılım.

### Genel değerlendirme (operatörün "holdout'ta korunursa ekonomik validation" koşulu için)
Hipotez yön olarak TRAIN+TEST'te korunuyor (p=0.0255) ama: **(a)** negative control benzer etkiyi rastgele zamanlarda da gösteriyor (cascade-spesifik değil olabilir), **(b)** day_trend_bps confounder'ı belirgin farklı, **(c)** aylık stabilite kusurlu (2026-06 ters yönde). **Bu, "holdout'ta temiz biçimde korundu" değil, "yönü tutuyor ama cascade-spesifik olduğu şüpheli, confounder-ayarlanmamış" bir sonuç** — ekonomik validation dalgası için otomatik yeşil ışık verilmedi; operatöre üç seçenek sunuluyor (bkz. rapor).

### Testler ve doğrulama
12 yeni test (`test_ami_research_w6rs_confirmation.py` 11/11: day_trend_bps hesabı, Wilson CI sınırları, permütasyon testi [ayrık/örtüşen gruplar/n=0], sentetik+gerçek-veri end-to-end, canonical-SQL yazım+idempotency; `test_ami_research_feature_gateway.py` +1: notional alanı). **Toplam AMI: 326/326 ✓** (314+12). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=17, idempotent).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0017 (W6-RS confirmation prereg+veri+feature_gateway notional). Yeni MD oluşturulmadı. Hiçbir bucket/route/observer davranışı değiştirilmedi. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım (confound-ayarlanmış reanaliz / ekonomik validation / farklı dalga) için operatör kararı bekleniyor.

## 59. BATCH-P6-009 — W6-RS CONFOUND-RESOLUTION WAVE (POST-HOC SENSITIVITY/SPECIFICITY): REGIME_DEPENDENT_CONTINUE_ACCUMULATING (2026-07-04, Sonnet 5)

**Active phase:** 6 (W6-RS confound-resolution) → **TAMAMLANDI**. **Bu bir POST-HOC SENSITIVITY/SPECIFICITY ANALİZİDİR — confirmatory kanıt olarak sunulmuyor** (operatör talimatı). RS_threshold=0, RS_lookback=1h, 24h outcome AYNEN dondu; hiçbir sweep yapılmadı. Ekonomik iddia/bucket promotion yok; bucket/route/observer değişmedi.

**Exact amaç:** RS>0'ın 24h REVERSAL ile ilişkisi, genel piyasa-rejimi etkisinden bağımsız olarak SELL-cascade anchor'larına ÖZGÜ ek bilgi taşıyor mu?

### Yöntem
`ami/research/w6rs_confound_resolution.py` — gerçek anchor'lar (n=243) + candidate-universe negative control (n=1985, aynı modelde) TEK bir logistic regression'da birleştirildi: `outcome ~ anchor_status + rs_group + anchor_status×rs_group(interaction) + day_trend_bps_z + session + month`. **notional joint modele girmedi** (control zamanlarında tanımsız — cascade olmayan bir anda "notional" kavramı yok; anchor-only confound olarak önceki dalgadan (W6-RS confirmation) aynen aktarıldı, tekrar hesaplanmadı). IRLS logistic regression numpy ile elle yazıldı (statsmodels ortamda yok). Belirsizlik: canonical-v1 `cycle_id` ile cluster block-bootstrap (n=2000, sabit seed) — anchor satırları cycle'a göre, control satırları her biri kendi kümesi olarak.

### Ana test: anchor_status × RS_group interaction
| Metrik | Değer |
|---|---|
| independent_cycle_n | **159** |
| interaction katsayısı (nokta tahmini) | **+0.397** (pozitif) |
| interaction bootstrap %95 CI | **(−0.28, 1.08) — sıfırı içeriyor** |
| kronolojik holdout (anchor-only RD) | TRAIN(n=170)=+16.7pp, TEST(n=73)=+7.6pp — yön tutarlı |
| aylık yön | Şub/Mar/Nis tutarlı; **Haz(n=26/76) VE Tem(n=10/3, berabere) uyumsuz** — kovaryat düzeltmesinden SONRA da açıklanamadı |
| overlap/positivity | day_trend_bps aralıkları makul örtüşüyor; EUROPE session anchor'larda ince (n=1/4) — dikkatli okunmalı |
| model uyarısı | `month_2026-07` katsayısı aşırı büyük (17.4) — o ay için olası ayrışma/instabilite, dürüstçe not edildi |

### Karar kuralları uygulaması
Operatörün 3 kuralı mekanik olarak uygulandı: (1) interaction CI sıfırı içeriyor → GENERAL_REGIME_FEATURE adayı olurdu, AMA (3) Haziran'ın yön-tersine-dönmesi kovaryat setiyle açıklanamadığı için **kural #3 önceliklendirildi**: **VERDICT = REGIME_DEPENDENT_CONTINUE_ACCUMULATING**. Ne "RS genel bir rejim özelliği, cascade'e özgü değil" (kesin ret) ne de "cascade'e özgü, ekonomik validation'a geç" (kesin onay) — **daha fazla veri (özellikle daha fazla ay/rejim) toplanana kadar kesin sınıflandırma YAPILMADI.**

### Testler ve doğrulama
10 yeni test (`test_ami_research_w6rs_confound_resolution.py`: IRLS'in bilinen sentetik ilişkiyi doğru kurtarması + gürültüde spurious katsayı üretmemesi, tasarım matrisi şekli, 3 karar-kuralı dalı + insufficient-sample dalı, thin-cell tespiti, cluster-bootstrap CI üretimi, gerçek-veri end-to-end+idempotency). **Toplam AMI: 336/336 ✓** (326+10). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=9, idempotent).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0018 (W6-RS confound-resolution prereg+veri). Yeni MD oluşturulmadı. Hiçbir bucket/route/observer değişmedi. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım (daha fazla veri birikince yeniden değerlendirme / OD-013-014-015 backlog / farklı dalga) için operatör kararı bekleniyor.

## 60. BATCH-P6-010 — W7A (STATE/STRUCTURE AGING + MARKET CLOCKS): DONDURULMUŞ PREREGISTRATION ÇALIŞTIRILDI, NÜANSLI/NULL SONUÇ (2026-07-04, Sonnet 5)

**Active phase:** 6 (W7A) → **TAMAMLANDI**. Checkpoint doğrulaması (SYSTEM_STATE §59, IMPLEMENTATION_PROGRESS_LEDGER son satırı, TEST_STATUS_LATEST, OPERATOR_DECISION_QUEUE, canonical `experiment_registry` son kaydı) repo ile birebir eşleşti; ardından `HISTORICAL_RESEARCH_WAVES.md`'de önkoşulu (P3+P5) zaten karşılanmış tek sıradaki dalga olarak **W7** operatöre önerildi ve onaylandı. Operatör ilk taslak kapsamı genel olarak onayladı, ardından kodlamadan önce **5 düzeltme** istedi (hepsi uygulandı, aşağıda).

### Operatörün 5 düzeltmesi (preregistration'a işlendi, kodlamadan ÖNCE dondu)
1. **Continuous age bucket cutpoint = TRAIN medyanı** (full-population medyan DEĞİL) — TEST ve candidate negative-control'a aynen uygulanıyor.
2. **Market-clock completeness gate:** anchor penceresi 60/60 geçerli 1m candle, baseline ≥24/30 geçerli 1h pencere, collector/gap durumunda veya baseline medyan=0 durumunda MISSING.
3. **Liquidation age:** canonical event-membership dışlaması TERCİH EDİLDİ ve UYGULANABİLDİ — `ami_events.event_end_ts_ms` (248/252 anchor'da mevcut, 4'ünde anchor_ts_ms'e fallback) `feature_gateway.fetch_events()`'e eklendi; generic isimler korundu (pre2h_ fallback ismi KULLANILMADI çünkü canonical dışlama gerçekten uygulanabildi).
4. **Exact maturity cutoff:** `MATURITY_CUTOFF_TS_MS = MAX(ami_candles.close_ts_ms)`, 24h ufku henüz oluşmamış anchor'lar dışlandı (excluded_no_horizon_data); ileride olgunlaşacak anchor'lar bu deneyin sayılarını YERİNDE DEĞİŞTİRMEYECEK, ayrı bir append-only follow-up experiment_id olarak kaydedilecek (süreç dokümante edildi, bu batch'te henüz gerekmedi).
5. **İnference planı dondu:** primary=independent-cycle cluster block-bootstrap risk-difference CI (n=2000); secondary=two-sided label-permutation (n=2000) + 9 test üzerinde Holm step-down adjustment.

### W7A preregistration + kapsam (dondurulmuş, `ami/research/w7a_state_structure_aging_market_clocks.py` docstring'i = freeze)
**Popülasyon/outcome:** W4'ten aynen (anchor_n=252, swing_24h path class, yeniden tanımlanmadı). **9 bağımsız feature/test** (FAM_SIGNAL_AGING_MARKET_CLOCK — "4 bileşen" 9 gerçek teste karşılık geliyor, operatörün kendi çerçevelemesi):
- **state_age** — `StateEngine._structure()` 1h TF dwell-time, cadence=1h, cap=168 adım (7 gün, performans gerekçeli), taksonomi: OK/LEFT_CENSORED (veri başlangıcı)/LEFT_CENSORED_AT_CAP (hesap sınırı)/MISSING (2h'den eski stale mark_price — orta-seri collector boşluğu).
- **swing_age/level_age/push_age** — 3 AYRI feature (birleştirilmedi), known_at_ts nearest-prior, MISSING = uygun nesne yok.
- **trade_count_clock/volume_clock/realized_vol_clock** — 3 AYRI feature, 60-candle (1h) pencere, 30 çakışmayan baseline penceresinin medyanına oranla (ratio>1.0=HIGH, sweep yok), completeness gate'leri (#2) her ikisine de uygulandı.
- **liq_age_same_direction (SELL) / liq_age_opposite_direction (BUY)** — 2 AYRI feature, 200K eşik (`_cascade()`'den yeniden kullanım, yeni fit değil), `event_end_ts_ms` ile canonical-cascade dışlama (#3).

**OD-016 eklendi:** whitepaper §61'in tam sinyal-lifecycle şeması (`signal_birth_ts`/`first_executable_ts`/`time_since_last_confirmation`) ve `book_update_age` Phase 8 forward-observer + order-book-continuity gerektiriyor — bu dalgada zorlanmadı, backlog'a alındı.

### Gerçek sonuçlar (E-W7A-STATE-STRUCTURE-AGING-MARKET-CLOCKS-001, canonical.sqlite)
total_anchor_n=252, **analyzed_n=243** (excluded_no_horizon_data=9 — W6RS-confirmation ile BİREBİR aynı neden/sayı, tutarlılık teyidi), **independent_cycle_n=159**. TRAIN medyanları (7 gün cap'e rağmen state_age=0ms — cascade anının kendisi StructurePhase değişimini büyük olasılıkla TETİKLİYOR, beklenen/mantıklı bir bulgu, bug değil; swing/level_age ~3dk, push_age ~4dk, liq_age_same ~98dk, liq_age_opposite ~231dk).

| Test | Holm-öncesi p | Holm-adjusted p | Bootstrap CI (95%) | Not |
|---|---|---|---|---|
| liq_age_same_direction | 0.0435 | **0.348** | (−0.0065, 0.2549) | Sıfıra çok yakın alt sınır, düzeltme sonrası anlamsız |
| realized_vol_clock | 0.0295 | **0.2655** | (0.027, 0.3531) | CI sıfırı dışlıyor ama Holm-adjusted anlamsız |
| diğer 7 test | ≥0.35 | ≥0.35 | hepsi sıfırı içeriyor | Ayırt edici değil |

**Sonuç: 9 testin HİÇBİRİ family-wise (Holm) anlamlı değil.** Dürüst null/nüanslı sonuç — hiçbir trade/PnL iddiası, bucket/route/observer promotion YAPILMADI.

### [2026-07-04, EK] Checkpoint kapanışı — operatör talimatı
Operatör W7A checkpoint'ini kapattı: 2 nominal (Holm-öncesi) p<0.05 sonucu (`liq_age_same_direction`, `realized_vol_clock`) post-hoc eşik/subgroup taramasıyla kurtarılmaya ÇALIŞILMADI. `classify_closure()` fonksiyonu eklendi (`ami/research/w7a_state_structure_aging_market_clocks.py`) — 9 testin her biri programatik olarak NULL/UNCONFIRMED_DESCRIPTIVE_LEAD/HOLM_SIGNIFICANT/UNDEFINED'e sınıflandırılıp `closure_classification` sonucu olarak canonical SQL'e yazıldı (idempotent, 2× doğrulandı, 17/17 sonuç satırı). Gerçek sınıflandırma: 7 NULL + 2 UNCONFIRMED_DESCRIPTIVE_LEAD (liq_age_same_direction, realized_vol_clock) — hiçbiri HOLM_SIGNIFICANT değil. Bu iki lead **kayıtlı ama kovalanmadı**; ileride yeni veri/rejim birikirse ayrı bir prereg ile yeniden açılabilir (yeni hipotez, mevcut testin eşiği değiştirilerek değil). 5 yeni test eklendi (classify_closure'ın 4 dalı + gerçek-veri closure kilidi). **Toplam AMI: 363/363 ✓** (358+5). **W7A checkpoint KAPALI.**

### Testler ve doğrulama
20 yeni test (`test_ami_research_w7a_state_structure_aging_market_clocks.py`: structural-object-age nearest-prior/missing, state-age 4 taksonomi dalı [StateEngine-backed synthetic mark_prices ile OK/LEFT_CENSORED/LEFT_CENSORED_AT_CAP/MISSING], liquidation-age direction-independence, market-clock 3 completeness-gate senaryosu + OK-ratio hesabı, Holm-adjust bilinen-değer testi + None-passthrough, permutation-test sınır davranışı, gerçek-veri end-to-end+idempotency) + 2 yeni (`test_ami_research_feature_gateway.py`: `event_end_ts_ms` mevcut/None). **Toplam AMI: 358/358 ✓** (336+22). `test_ami_warehouse_registry_seed.py`'de OD-016 eklenmesiyle bozulan 2 sabit-sayı assertion'ı (15→16) düzeltildi (satır sayısı sabit 7, yalnız içerik). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=16, idempotent) — tüm koşum **~4.7 saniyede** tamamlandı (SQLite indeksli sorgular beklenenden çok daha hızlı, performans endişesi gerçekleşmedi).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0019 (W7A prereg+veri+feature_gateway event_end_ts_ms). `OPERATOR_DECISION_QUEUE.md` OD-016 (yeni, sinyal-lifecycle+book_update_age backlog). Yeni MD oluşturulmadı. Hiçbir bucket/route/observer değişmedi. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım (W7'nin geri kalanı [signal aging'in Phase-8-bağımlı kısmı, OD-016] / W8-12 / OD-013-015 backlog / RS confound re-evaluation) için operatör kararı bekleniyor.

## 61. BATCH-P6-011 — W8-12 READ-ONLY AUDIT + W10a (MULTI-TF STRUCTURAL CONFLICT): NULL SONUÇ (2026-07-04, Sonnet 5)

**Active phase:** 6 (W10a) → **TAMAMLANDI**. Önce `HISTORICAL_RESEARCH_WAVES.md`'deki W8-12'nin dependency/graveyard/coverage durumu read-only incelendi (kod yazılmadan), sonra tek FULLY-buildable parça (W10'un multi-TF conflict yarısı) preregister edilip operatörün 7 düzeltmesiyle dondurulup çalıştırıldı.

### W8-12 audit sonucu
| Dalga | Önkoşul | Mezarlık | Sınıflandırma |
|---|---|---|---|
| W8 (competing-risk hold) | P7 — **kurulu değil** (`ami/warehouse/schema.py`'de hold/exit/timing tablosu yok) | YÜKSEK: #3 tight-stop, #4 partial-exit, #10 loser-time-stop, #13 MFE50 (retry_condition literal "lifecycle engine" istiyor), #16/#18/#21 | **GRAVEYARD_COLLISION** + NOT_IMPLEMENTED |
| W9 (stop taxonomy+re-entry) | P7 — kurulu değil | #3/#10 (stop), #19 (re-entry, retry_condition="yalnız OD-008 onayıyla" — OD-008 hâlâ OPEN) | **GRAVEYARD_COLLISION** + NOT_IMPLEMENTED + OD-008 bekliyor |
| W10 (transitions + multi-TF conflict) | P7 (nominal); multi-TF parçası Phase-1 StateEngine'de zaten var | "multi-TF conflict" için mezarlıkta 0 sonuç; transition whitepaper §54.5'te Phase-8 forward-dashboard sayfası | **PARTIALLY_BUILDABLE**: multi-TF=BUILDABLE_NOW, transition=BLOCKED_BY_DATA (popülasyon %100 SELL) + NOT_IMPLEMENTED |
| W11 (proxy reconstruction) | P3 identity — **kurulu** (R-09 test edilmiş) | Yok, ama `ami_events` 252/252 REAL_LIQUIDATION, **0 PROXY_\*** — karşılaştırılacak veri yok | **BLOCKED_BY_DATA** + NOT_IMPLEMENTED |
| W12 (action-value) | W4+W8 — W4 var, **W8 yok** | Whitepaper zaten "action-value output cannot create an order" diyor (mutation-tested) | **NOT_IMPLEMENTED** (kırık bağımlılık) |

**Sonuç:** W8-12'nin hiçbiri tam BUILDABLE_NOW değil; tek istisna W10'un multi-TF conflict yarısı.

### W10a preregistration (dondurulmuş, operatörün 7 düzeltmesiyle — `ami/research/w10a_multi_tf_structural_conflict.py` docstring'i = freeze)
- **Popülasyon/outcome:** W1/W4'ten aynen (anchor_n=252, swing_24h path class).
- **TF çifti:** 1h/4h — OD-003'ün cycle_resolver'ından birebir alındı, yeni seçim değil.
- **Direction mapping:** `StateEngine._structure()`'ın MEVCUT/DEĞİŞMEMİŞ `direction` (UP/DOWN/FLAT) formülünden birebir alıntı — yeni icat yok.
- **[Düzeltme 1]** 5 ham hücre (UP_UP/DOWN_DOWN/UP_DOWN/DOWN_UP/NEUTRAL) veri-seviyesinde AYRI korunuyor, sonuç görülmeden AGREE olarak erken birleştirilmiyor.
- **[Düzeltme 2]** TEK preregistered primary hipotez: CONFLICT(UP_DOWN∪DOWN_UP) vs AGREEMENT(UP_UP∪DOWN_DOWN) — 5-hücreli tablo yalnız descriptive/mekanizma raporu, 5 ayrı p-value üretilip family gizlice genişletilmiyor.
- **[Düzeltme 3]** NEUTRAL primary contrast'a HİÇ eklenmiyor (N'i ne olursa olsun); kendi tanımlayıcı kategorisinde kalıyor, N<20 ise INSUFFICIENT_SAMPLE, başka grupla birleştirilmiyor.
- **[Düzeltme 4]** Known-at güvenliği explicit testle kanıtlandı: `_structure()`→`_ret_bps()`→`_px()` candle değil, `mark_prices`'a `ts_ms<=ts` tick-tabanlı sorgu yapıyor — `test_direction_classification_is_known_at_safe` anchor'dan sonraki uç bir mark_price satırının anchor'daki yönü DEĞİŞTİRMEDİĞİNİ kanıtlıyor.
- **[Düzeltme 5]** Mapping formülü docstring'de birebir alıntılandı, dondu.
- **[Düzeltme 6]** Exact maturity cutoff (W6RS/W7A paterni) + kronolojik 70/30 + independent-cycle N + candidate-universe negative control korundu.
- **[Düzeltme 7]** OD-017 eklendi (LONG↔SHORT transition yarısı backlog'da: BLOCKED_BY_DATA + NOT_IMPLEMENTED).

`ami/research/w7a_state_structure_aging_market_clocks.py:cluster_bootstrap_risk_difference()`'e `label_high`/`label_low` parametreleri eklendi (geri-uyumlu, varsayılan HIGH/LOW değişmedi, W7A etkilenmedi) — W10a'nın CONFLICT/AGREEMENT etiketleriyle aynı bootstrap kodunun tekrar kullanılması için.

### Gerçek sonuçlar (E-W10A-MULTI-TF-STRUCTURAL-CONFLICT-001, canonical.sqlite)
analyzed_n=243, independent_cycle_n=159. Ham hücreler: DOWN_DOWN baskın (146/243, %60 — cascade popülasyonunun ayı-eğilimli doğasıyla tutarlı), UP_UP(18)/NEUTRAL(57) orta, **UP_DOWN(n=3) ve DOWN_UP(n=19) INSUFFICIENT_SAMPLE** (dürüstçe raporlandı, birleştirilmedi). **Primary contrast:** CONFLICT(n=22, REVERSAL=59.1%) vs AGREEMENT(n=164, REVERSAL=62.2%) — risk-difference=−0.031, permütasyon p=0.8275, bootstrap CI=(−0.2717, 0.2094) sıfırı geniş marjla içeriyor — **null sonuç**. Kronolojik 70/30'da CONFLICT bucket'ı zaten yetersiz-örneklem (train n=17, test n=5 — stabilite kontrolü kendisi underpowered, dürüstçe flag'lendi). Candidate-universe negative control'de de anlamlı fark yok (p=0.1245). **Hiçbir ekonomik iddia veya bucket/route/observer promotion YAPILMADI.**

### Testler ve doğrulama
15 yeni test (`test_ami_research_w10a_multi_tf_structural_conflict.py`: 5 hücrenin tüm sınıflandırma dalları, primary-bucket exclusivity [NEUTRAL asla AGREEMENT/CONFLICT'e düşmüyor], insufficient-sample'ın birleştirilmediği kilidi, known-at-safety [StateEngine-backed, gelecekteki mark_price yönü değiştirmiyor], gerçek-veri end-to-end+idempotency) + 1 yeni (`test_ami_research_w7a_...py`: cluster_bootstrap custom label parametrizasyonu). **Toplam AMI: 379/379 ✓** (363+16). `test_ami_warehouse_registry_seed.py` OD-017 için güncellendi (16→17, satır sayısı sabit 7). Protected diff: yalnız pre-existing baseline. Gerçek veri 2× çalıştırıldı (registry=1/results=9, idempotent).

### Kalıcı kayıtlar
`MIGRATION_LOG.md` M-0021 (W10a prereg+veri+w7a additive parametre). `OPERATOR_DECISION_QUEUE.md` OD-017 (yeni, LONG↔SHORT transition backlog). Yeni MD oluşturulmadı. Hiçbir bucket/route/observer değişmedi. Model değişmedi, Sonnet 5 devam ediyor — sıradaki adım (W8/W9/W11/W12'nin önkoşul-bağımlılıkları çözülünce yeniden değerlendirilmesi / OD-013-017 backlog / RS confound re-evaluation / farklı bir Phase 6+ dalgası) için operatör kararı bekleniyor.

### [2026-07-04, EK] W10a checkpoint kapanışı
Primary contrast zaten temiz null (p=0.8275, bootstrap CI geniş marjla sıfırı içeriyor) — W7A'daki gibi ambiguous nominal-p bulgusu yok, ek `classify_closure` mekanizması gerekmedi. **W10a checkpoint KAPALI.** Operatör talimatı gereği küçük historical-descriptive dalga aranması bu noktada DURDURULDU; sıradaki iş Phase 7-8 minimum altyapı tasarım denetimidir (aşağı).

## 62. PHASE 7-8 MINIMUM TIMING/LIFECYCLE INFRASTRUCTURE — READ-ONLY DESIGN AUDIT (2026-07-04, Sonnet 5)

**Active phase:** 7-8 ön-tasarım (kod YAZILMADI, live/order/observer ÇALIŞTIRILMADI). Amaç: W8/W9/W10-transition/W12'yi açacak en küçük ortak altyapıyı tanımlamak. Operatör onayı bekleniyor — bu bölüm yalnız dependency map + implementation planı.

### [2026-07-04, EK — operatör düzeltmeleri] Protected diff / binding / sınıflandırma kayıtları

**1. Protected diff durumu — düzeltildi:**
```
PRE_EXISTING_PROTECTED_DIRTY_STATE = TRUE
SESSION_DELTA_ON_PROTECTED_FILES   = ZERO
```
Bu "clean" DEĞİL — konuşma başlangıcından ÖNCE zaten kirli (dirty) 2 dosya vardı; bu oturum onlara SIFIR ek değişiklik yaptı. Kanıt (audit başlangıcı ≈ konuşma başlangıcı, audit sonu = şimdi):

| Kontrol | Konuşma başlangıcı (ilk gitStatus) | Şimdi (§62 sonrası) |
|---|---|---|
| `git status --short` (tam repo) | `M tools/s34_cascade_navigation_dashboard.py`, `M tools/s34_realtime_shadow_runner.py`, + aynı untracked MD/dizin listesi | **BİREBİR AYNI** — ek satır yok, eksik satır yok |
| `tools/s34_realtime_shadow_runner.py` diff | 1135 satır değişik (pre-existing) | Aynı: `1 file changed, 1087 insertions(+), 48 deletions(-)` (=1135) |
| `tools/s34_state_machine_live_executor.py` | untracked (`??`) | Aynı, hâlâ untracked; sha256=`a8067f2c2575e60049382671fa955878b442c59178b28d6ab8cbef44a2804be0` (bu oturumdan sonraki referans hash — ileriki batch'lerde bu değerle karşılaştırılacak) |
| `tools/s34_realtime_shadow_runner.py` sha256 (bu oturum sonu referansı) | — | `a326a5c43b5c4095840ac9fc0f89a22f15f8e4d9aa75c9946c6ea9991d165e27` |
| `execution/`, `risk/`, `brain/`, `.env` | `git status --short` boş | Aynı, boş (0 satır — ne tracked-modifikasyon ne yeni untracked dosya) |

**Exact changed-file list (bu oturumun KENDİ yazdığı dosyalar, protected OLMAYAN):** `ami/research/w7a_state_structure_aging_market_clocks.py`, `ami/research/w10a_multi_tf_structural_conflict.py`, `ami/research/feature_gateway.py` (additive kolon), `ami/warehouse/registry_seed.py`'nin OKUDUĞU `OPERATOR_DECISION_QUEUE.md`, `tests/test_ami_research_w7a_*.py`, `tests/test_ami_research_w10a_*.py`, `tests/test_ami_research_feature_gateway.py`, `tests/test_ami_warehouse_registry_seed.py`, `SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md`/`MIGRATION_LOG.md` (dokümantasyon), `data/ami/canonical.sqlite` (deney verisi). **Bunların hiçbiri protected listede değil.**

**2. İki aktif binding — kayıt altına alındı:**
```
PRE_EXISTING_BINDINGS_FOUND      = 2
STARTED_DURING_AUDIT             = 0
STOPPED_DURING_AUDIT             = 0
CONFIG_CHANGED_DURING_AUDIT       = 0
ROWS_WRITTEN_BY_AUDIT_ACTIONS     = 0
```
(§62 denetimi tamamen read-only idi; `data/ami/research.sqlite:forward_bindings`/`processed_trades` bu denetim sırasında yalnız SELECT ile okundu, hiç INSERT/UPDATE yapılmadı — doğrulama: `processed_trades` toplam satır sayısı hâlâ 0.)

| Alan | E-HOUR17-FWD-001 | E-CONVCOMP-FWD-001 |
|---|---|---|
| knowledge_id (setup ID) | K-S34-HOUR17-001 | K-S34-MECH-COMPOSITE-001 |
| signal (family) | LONG_HOUR17_HOLD6H | LONG_HOUR17_COMPOSITE (conviction≥4) |
| spec_hash | 62861d5c6bf98581 | 15d4fc5c2c1a2038 |
| dataset_hash | s34-2026H1 | s34-2026H1 |
| code_ref | tools/research_s34_silence_predictor.py | tools/s34_mechanism_taxonomy.py |
| execution_model | mark_fill_fee5bps | mark_fill_fee5bps |
| process/entrypoint | `python -m ami.run_forward_pipeline` (cron/oturum-başına, idempotent — `ensure_binding()` "already_bound" kontrolü) | aynı |
| output tables | `data/ami/research.sqlite:forward_bindings,processed_trades,experiments` + `data/ami/knowledge.sqlite:knowledge,audit_log` + `reports/research/s34/AMI_FORWARD_EVIDENCE.md` | aynı |
| current observation mode | OBSERVATION_ONLY (governor yalnız KO status'ünü promote/demote edebilir; hiçbir LIVE/SIZING permission asla verilmiyor) | aynı |
| order-capable dependency | **YOK** — `ami/lifecycle/engine.py`/`ami/research/forward_pipeline.py`/`ami/run_forward_pipeline.py` import grafiği doğrulandı: `execution/`, `risk/`, `brain/`, order_router/entry_loop/position_manager hiçbir yerde import edilmiyor; `.env`/API-key okuması yok | aynı |
| FORWARD_N (şu an) | 0 (processed_trades: 0 satır, KnowledgeStatus=FORWARD_VALIDATING, henüz promote edilmedi) | 0 (KnowledgeStatus=HOLDOUT_VALIDATED) |

**3. Mevcut lifecycle altyapısı sınıflandırması — düzeltildi:**
```
EXISTING_REUSABLE_SKELETON               = TRUE   (TradeLifecycleState, classify_lifecycle_path,
                                                    ForwardEvidencePipeline R1-R6, EpistemicGovernor gates)
NOT_YET_CANONICAL_FOR_AMI_EVENTS_AMI_CYCLES = TRUE (bu altyapı LONG_HOUR17_*/LONG_SILENCE sinyallerine
                                                    bağlı; Phase 1-6'nın ami_events/ami_cycles [SELL-cascade,
                                                    252 anchor/167 cycle] popülasyonuna HİÇ bağlanmadı)
```
**Açık düzeltme:** `LONG_HOUR17_*`/`LONG_SILENCE` binding'lerinin var olması, Phase 1-6 `ami_events`/`ami_cycles` popülasyon kapsamının VAR OLDUĞU anlamına GELMEZ — bunlar TAMAMEN AYRI, hiç kesişmeyen iki veri kümesidir (biri whitepaper-numaralı eski çalışmanın LONG-route ledger alt-kümesi, diğeri Master-Roadmap'in SELL-cascade canonical event/cycle kimliği). Bu denetimin 0. bölümündeki "büyük kısmı zaten var" ifadesi yalnız KOD/İSKELET seviyesinde geçerlidir (vocabulary, gate mekanizması, append-only/idempotent desen) — POPÜLASYON/VERİ seviyesinde geçerli DEĞİLDİR.

### 0. KRİTİK ÖN-BULGU: bu altyapının büyük kısmı ZATEN VAR (whitepaper Phase 0-9 numaralandırması, `docs/ami/AMI_ROADMAP.md`)
Bu denetim sırasında, Master Roadmap'in (Fable 5, 2026-07-03) "mevcut doğrulanmış çekirdek (ami/ 119 test) üzerine eklemeli" talimatının atıfta bulunduğu ÖNCEKİ whitepaper-numaralı çalışma bulundu — P1-001'den W10a'ya kadar hiçbir batch bunu okumamıştı çünkü Phase 1-6 işi tamamen `ami_events`/`ami_cycles` (SELL-cascade popülasyonu) üzerine kuruluydu. Bulgular:

| Dosya | Whitepaper-fazı | Durum | Bu denetim için önemi |
|---|---|---|---|
| `ami/lifecycle/engine.py` | Faz 3 "Trade lifecycle" ✅ TAMAM | `TradeLifecycleState` taksonomisi (`ami/enums.py`: OPEN/HEALTHY/ACCELERATING/STALLING/WEAKENING/EXHAUSTED/RECOVERING/LOCKED/REVERSING/INVALIDATED/CLOSED) + `classify_lifecycle_path()` (1m PnL path → state dizisi, TAMAMEN descriptive, order üretmiyor) + `replay_shadow_ledger()` (MFE-milestone istatistiği: A_continue/B_breakeven/C_negative/D_time_pos) | Section 2/3'ün "state-transition ledger" + "path/progress" vocabulary'si BURADA ZATEN VAR — yeniden icat edilmeyecek |
| `ami/research/forward_pipeline.py` (`ForwardEvidencePipeline`) | Faz 4 Research OS ✅ TAMAM | R1-R6 kuralları KOD OLARAK zaten uyguluyor: R1=PRE_FREEZE reddi (FORWARD_N=0'dan başlar), R3=processed_trades PK dedup (APPEND_ONLY+idempotent), R5=governor'a yalnız BAŞVURU yapar izin VERMEZ (OBSERVATION_ONLY), R6=execution/config/.env/leverage/sizing'e hiç dokunmaz (fiziksel ayrım zaten sağlanmış). **CANLI:** 2 aktif frozen binding (E-HOUR17-FWD-001, E-CONVCOMP-FWD-001), `ami/run_forward_pipeline.py` ile cron'da idempotent koşuyor | Section 4'ün TAMAMI zaten kod olarak var ve ÇALIŞIYOR — Phase 8A "yeni bir observer inşa etmek" değil, BUNU genişletmek |
| `ami/governance/governor.py` (`EpistemicGovernor`) | Faz 5 ✅ TAMAM | `PermissionDecision` (GRANTED/DENIED/SHADOW_ONLY/OBSERVE_ONLY), `promote/demote` kodlu kapılarla (PROMOTION_GATES), circuit breaker, belief revision | forward_pipeline'ın zaten yönlendiği kontrol düzlemi — yeni bir gate icat edilmeyecek |
| `ami/states/structure.py` | Faz 2 (kısmi) | Tek-TF faz-geçiş matrisi (dwell/transition olasılığı) — W10a'nın "1h vs 4h anlık conflict" kavramından TAMAMEN FARKLI, çakışma yok | Bilgi amaçlı, W10a'yı etkilemiyor |
| `ami/identity/shadow_ledger_ingest.py` | Master Roadmap Faz 3 | `event_end_ts_ms` = MAX(bu anchor'a bağlı TÜM logical trade'lerin exit_ts'i) — yani "cascade'in kendi sonu" DEĞİL, "son ilişkili trade'in kapanış anı" (W7A'da bu alan liquidation-age dışlama sınırı olarak kullanılmıştı; sonucu geçersiz kılmıyor ama terminolojik netlik burada düzeltiliyor). `feature_available_ts_ms` kolonu ŞEMADA ZATEN REZERVE (şu an hep NULL) | Section 1'in first_known_ts/first_executable_ts'i için YENİ kolon değil, MEVCUT rezerve kolonun doldurulması |

**Sonuç:** İstenen "minimum ortak altyapı"nın kavramsal iskeleti (state vocabulary, observation-only gate, append-only forward loop, crash-safety, no-order garantisi) **ZATEN VAR VE CANLI**. Eksik olan: (a) bu parçalar `ami_events`/`ami_cycles` (SELL-cascade, Phase 1-6) popülasyonuna DEĞİL, `LONG_HOUR17_*`/`LONG_SILENCE` adlı sinyallere bağlı; (b) zenginleştirilmiş timing alanları (signal_birth_ts, first_known_ts, invalidation_ts, reason code, MFE/MAE/progress, volatility-normalized progress) HİÇBİR popülasyonda henüz hesaplanmıyor; (c) **`ami/lifecycle/engine.py` ve `ami/research/forward_pipeline.py`'ın SIFIR dedike unit testi var** (`tests/` içinde ne `LifecycleEngine` ne `ForwardEvidencePipeline` referansı bulundu) — canlı ve üretimde ama test edilmemiş; bu iki modülü genişletmeden önce KAPATILMASI gereken bir boşluk.

### 1. Canonical lifecycle identity
| Alan | Kaynak | Sınıflandırma |
|---|---|---|
| `source_event_id` | `ami_events.event_id` (var) | REUSE |
| `independent_cycle_id` | `event_cycle_membership.candidate_cycle_key` (canonical-v1, var) | REUSE |
| `signal_id` | YENİ — 7A'da event_id ile 1:1 (basitleştirilmiş kapsam; çoklu-sinyal-per-event zenginleştirmesi bilinçli olarak ERTELENDİ, kapsam patlamasını önlemek için) | NEW (dar kapsamlı) |
| `setup_id`/`version` | YENİ — immutable-versioned (candle/cycle/level_definition_version paterniyle aynı disiplin: mutasyon yok, yeni version) | NEW |
| `signal_birth_ts` | 7A'da = `anchor_ts_ms` (sinyal=anchor 1:1 basitleştirmesiyle) | REUSE (aliased) |
| `first_known_ts`/`first_executable_ts` | `ami_events.feature_available_ts_ms` — **kolon zaten var, şu an hep NULL** | POPULATE existing reserved column |
| `last_confirmation_ts` | YENİ kavram, hiçbir yerde tanımlı değil — 7A'da tanımlanacak (W7A'nın structural/liquidation-age çalışmasından ilham alınabilir ama fit edilmiş bir eşik DEĞİL, tanım netleşmeden dondurulmayacak) | NEW |
| `invalidation_ts` | YENİ, `terminal_ts`'den AYRI (ör. stop/iptal ile kapanış ile "hiç mi tetiklenmedi" farkı) | NEW |
| `terminal_ts` | `event_end_ts_ms` (mevcut) — ama gerçek anlamı "son ilişkili trade kapanışı", cascade sonu DEĞİL (yukarıdaki bulgu) | REUSE (anlamı netleştirilerek) |
| `lifecycle_status` | `ami/enums.py:TradeLifecycleState` — **birebir reuse, yeni enum icat edilmeyecek** | REUSE (verbatim) |
| reason codes | YENİ, küçük kontrollü sözlük — ham shadow ledger JSONL'deki gerçek alanlardan (fabrikasyon yok) türetilecek, 7A'da envanteri çıkarılacak | NEW (veri-türetilmiş) |

### 2. State-transition ledger
`prev_state`/`new_state` = `TradeLifecycleState` (reuse). `transition_ts` = `classify_lifecycle_path()`'ın MEVCUT algoritmasıyla (yeniden icat değil) belirlenen durum-değişim anı. `known_at_ts` = `transition_ts`'in kendisi (yalnız GEÇMİŞ mark_price path'i kullanılarak hesaplandığı için doğal olarak point-in-time-safe — W7A'nın known-at-safety ispat paternine aynen tabi olacak). **Append-only + idempotency:** `experiment_results`'ın DELETE+reinsert snapshot paterni DEĞİL — `event_cycle_membership`'in append-only paterni (satırlar asla silinmez/üzerine yazılmaz) + `UNIQUE(signal_id, transition_ts)` DB-seviyesi kısıtı.

### 3. Path/progress observations
MFE-milestone kavramı `replay_shadow_ledger()`'da ZATEN VAR (A_continue/B_breakeven/C_negative/D_time_pos) — yeni bucket isimleri icat edilmeyecek, genişletilecek. `state_age` = W7A'nın `compute_state_age()`'i DOĞRUDAN reuse (yeni bir build değil, mevcut fonksiyonun lifecycle-observation cadence'ine genelleştirilmesi). Volatility-normalized progress = W7A'nın market-clock medyan-normalizasyon yöntemi reuse edilecek (yeni normalizasyon şeması icat edilmeyecek). Cancellation/stop/hold/re-entry descriptive label'ları = `TradeLifecycleState` + `Action` enum'unun OBSERVE/WAIT/NO_TRADE/HOLD tier'ı (zaten `ACTION_REQUIRED_PERMISSION`'da RESEARCH_ONLY/SHADOW_ALLOWED'a bağlı, LIVE_ALLOWED'a asla değil) — **"hiçbir label order üretmemeli" zaten kod-seviyesinde bu enum-permission eşlemesiyle garanti altında**; Phase 8B'de bunu kanıtlayan bir mutation testi eklenecek (zorunlu kabul kriteri).

### 4. Forward observer sınırı
OBSERVATION_ONLY/NO_ORDER/APPEND_ONLY/FORWARD_N=0/crash-restart-idempotency — **hepsi `ami/research/forward_pipeline.py`'da ZATEN KOD OLARAK VAR VE CANLI ÇALIŞIYOR** (R1-R6, yukarıya bakınız). Fiziksel/mantıksal ayrım da zaten sağlanmış: pipeline yalnız `reports/shadow/s34_state_machine_shadow.jsonl`'ı OKUR (korunan executor'ın YAZDIĞI, hiç değiştirilmeyen dosya) ve yalnız `data/ami/research.sqlite`/`knowledge.sqlite`'a yazar — `execution/`/`risk/`/`brain/`/`.env`'e sıfır bağımlılık. Phase 8A'nın görevi **yeni bir observer icat etmek DEĞİL, bu pipeline'ı canonical `ami_events`/`ami_cycles` popülasyonunu da kapsayacak şekilde GENİŞLETMEK.**

### 5. Historical backfill politikası
**Deterministik/historical-safe (7A'da backfill edilebilir):** signal_birth_ts/source_event_id/independent_cycle_id (zaten var), lifecycle_status dizisi+transition_ts (mark_price path replay, deterministik), state/structural/liquidation-age+market-clock (W7A, zaten inşa+test edilmiş), MFE/MAE/progress (mark_price path replay, deterministik).
**FORWARD_ONLY (historical'da uydurulmayacak):** first_known_ts/first_executable_ts eğer "canlı sistemin kendi feature pipeline'ının bu sinyali İLK KEZ ne zaman hesaplayıp açığa çıkardığı" olarak tanımlanırsa — bu, çalışan sistemin kendi gecikme/zamanlama özelliğidir, sonradan "sanki o zaman bilseydik" diye icat edilemez, FORWARD_ONLY işaretlenecek. Aynı şekilde book_ticker-türevi zamanlama (OD-016'nın `book_update_age`'i) — book_ticker'ın tarihsel süreklilik/kapsamı bu denetimde ÖLÇÜLMEDİ (büyük tablo, `COUNT(*)` 60sn'de bile tamamlanmadı — bkz. §6), FORWARD_ONLY kalacak ta ki ayrı bir kapsam-denetimi yapılana kadar.
**Real vs proxy:** `SourceQuality` enum + `assert_not_pooled` guard'ı AYNEN reuse — yeni lifecycle tabloları da AYNI (veya onu genişleten) guard'dan geçecek, paralel bir "populasyon-saflığı" kontrolü icat edilmeyecek.

### 6. Veri bağımlılıkları
| Feed | Kapsam (bu denetimde doğrulandı) | Sınıflandırma |
|---|---|---|
| mark_prices/agg_trades/liquidations (ETH, BTC mark) | ~2026-02-15→bugün sürekli | READY |
| ami_candles/swings/levels/pushes | tam-tarih backfill edilmiş (M-0012/M-0013) | READY |
| open_interest | ETH: 75 günlük boşluk (OD-012); BTC/SOL: ~1.5-3 gün pencere | BLOCKED_BY_DATA (kısmi) |
| funding_rates | ÖLÜ, canlı üretici yok (2026-04/05'ten beri, OD-006 hâlâ OPEN) | BLOCKED_BY_DATA |
| book_ticker | Tablo var, `engine.py:FEED_LIMITS` 5dk-tazelik canlı tick feed'i olarak kullanıyor; **tam tarihsel kapsam/süreklilik bu read-only denetimde ÖLÇÜLMEDİ** (`SELECT COUNT(*)` 60 saniyede bile tamamlanmadı — pahalı tam-tablo taraması, ayrı küçük bir 7A alt-görevi olarak planlanacak, burada fabrike edilmedi) | COVERAGE_UNKNOWN → 7A'da çözülmeden `book_update_age` NOT_IMPLEMENTED kalır |
| vol_state | Tablo var ama önceki oturum notuna göre ÖLÜ/kırık, rv-proxy fix restart bekliyor (`project_divers_signal_jul2026` hafıza kaydı) | BLOCKED_BY_DATA (restart+doğrulama şartı) |
| `reports/shadow/s34_state_machine_shadow.jsonl` | Canlı, korunan executor tarafından sürekli yazılıyor; salt-okunur kaynak | READY (read-only) |

### 7. Minimum implementation sırası
| Faz | Kapsam | Yeni/değişen dosyalar | Test | Kabul kriteri |
|---|---|---|---|---|
| **7A** | Şema (v7→v8) + yalnız deterministik historical-safe alanlar; **signal_id=event_id 1:1 basitleştirmesiyle** (çoklu-sinyal zenginleştirmesi ERTELENDİ) | `ami/warehouse/schema.py` (+`ami_signal_lifecycle`, `+ami_lifecycle_transitions`, `+ami_lifecycle_path_observations`); YENİ `ami/lifecycle/canonical_backfill.py` (reuse: `classify_lifecycle_path`, W7A `compute_state_age`/`compute_structural_object_age`, W4 `compute_path_returns`) | schema round-trip, known-at-proof (W7A paterni), idempotency, backfill-vs-`replay_shadow_ledger` tutarlılık spot-check | 0 protected-diff; tüm known-at testleri geçer; backfill sayıları idempotent |
| **7B** | Timing/path metrics engine (MFE/MAE/progress/vol-normalized progress) | YENİ `ami/lifecycle/path_metrics.py` (reuse: MFE-milestone taksonomisi + W7A market-clock normalizasyonu) | MAE eklenmiş MFE hesaplarının sentetik path'te doğruluğu; **AYNI `LONG_HOUR17_*` alt-kümesinde `replay_shadow_ledger()` ile çapraz-doğrulama (eşleşmezse sessiz yeniden-tanım şüphesi)** | Çapraz-doğrulama eşleşir; 0 protected-diff |
| **8A** | Gözlem-yalnız forward lifecycle observer (`forward_pipeline.py` GENİŞLETME, YENİDEN İNŞA değil) | `ami/research/forward_pipeline.py`'a genişletme VEYA aynı binding paternini reuse eden kardeş modül; **ÖNCELİKLE mevcut `LifecycleEngine`/`ForwardEvidencePipeline` için EKSİK test kapsamı kapatılacak** (yeni koddan önce) | (a) eksik testlerin geriye-dönük eklenmesi, (b) genişletme testleri, (c) mevcut 2 canlı binding'in davranışının DEĞİŞMEDİĞİni kanıtlayan regresyon testi, (d) Action/Permission sınır-mutasyon testi (order asla üretilmiyor) | Mevcut 2 binding etkilenmez; yeni binding N=0'dan başlar; governor circuit-breaker testi geçer |
| **8B** | Validation/health/dashboard | `governor.check_data_health()` yeni feed'lere bağlama; salt-okunur dashboard sayfası (mevcut `tools/s34_live_chart.py` paterni) | circuit-breaker trip/reset testi; dashboard'ın order-yeteneği OLMADIĞININ testi | Dashboard read-only kanıtlı; sağlık izleme canlı |

**→ Ancak 8B'nin kabul kriterleri karşılandıktan SONRA** W8/W9/W10-transition/W12 yeniden değerlendirilebilir — ve o zaman bile otomatik açılmaz: her biri kendi mezarlık/OD kontrolünden TEKRAR geçmek zorunda (altyapının varlığı mezarlık disiplinini geçersiz kılmaz, Protocol §5).

**Tahmini batch sayısı:** 7A (1-2), 7B (1), 8A (2 — eksik test kapatma + genişletme ayrı), 8B (1-2). **Toplam ~6-8 batch**, W8/W9/W10-transition/W12'yi yeniden açmadan önce.

### 8. Güvenlik ve test planı
- **Migration:** additive-only (`CREATE TABLE IF NOT EXISTS`, v7→v8), rollback = yalnız yeni tabloları sil (ami_events/ami_cycles/experiment_* dokunulmaz).
- **Schema constraints:** `PRAGMA foreign_keys=ON` (zaten aktif) + yeni tablolarda `UNIQUE` doğal anahtar (append-only'yi DB-seviyesinde zorlar, yalnız kod disiplinine güvenilmez).
- **Known-at/no-lookahead:** her yeni alan kategorisi için W7A'nın `test_direction_classification_is_known_at_safe` paterninde EXPLICIT bir kanıt testi — zorunlu kabul kriteri.
- **Duplicate/restart/idempotency:** `forward_pipeline`'ın PK-dedup paterni reuse; her yeni tabloda 2× koşum testi (mevcut konvansiyon).
- **Mutation tests proving no order creation:** YENİ — lifecycle_status/reason-code'dan erişilebilir her Action/Permission kombinasyonunun `governor.promote()`'un TAM kapılarından geçmeden asla RESEARCH_ONLY/OBSERVER_ALLOWED üstüne çıkmadığını kanıtlayan sınır testi.
- **Protected diff gate:** her batch sonunda `git status --short -- tools/s34_state_machine_live_executor.py .env execution/ risk/ brain/ tools/s34_realtime_shadow_runner.py` sıfır fark göstermeli — bu track için EXPLICIT kabul kriteri (risk daha yüksek olduğundan).
- **Risk register:** (1) `LifecycleEngine`/`ForwardEvidencePipeline` canlı ama test edilmemiş — genişletmeden önce kapatılmalı; (2) book_ticker/vol_state veri kalitesi belirsiz/ölü — bazı alanları NOT_IMPLEMENTED'da tutuyor; (3) signal_id'nin 7A'da event_id'ye 1:1 basitleştirilmesi kapsamı daraltıyor (bilinçli, ama çoklu-sinyal zenginleştirmesi gelecekte AYRI bir prereg gerektirecek); (4) korunan bileşen sürüklenmesi riski (diff-gate ile azaltıldı); (5) whitepaper §54.6'nın önerdiği `ami_forward_*` tablo listesi bir REFERANS/kavram sözlüğü olarak kullanıldı, birebir sözleşme değil — uyuşmazlık kendi başına engelleyici değil, belgelenmiş bir takdir kararı.
- **Backlog eşlemesi:** OD-008 (BAD_TIMING) — 8B sonrası bile KENDİ ayrı operatör onayını gerektirir. OD-013/014/015 — bu hat'tan BAĞIMSIZ, istenirse paralel ilerleyebilir. **OD-016 (sinyal-lifecycle tam şema+book_update_age) — DOĞRUDAN bu hat tarafından ele alınıyor**, 7A/8A tamamlanınca yeniden gözden geçirilmeli. OD-017 (LONG↔SHORT transition) — 8B sonrası bile BLOCKED_BY_DATA olarak kalır (altyapı LONG-anchor popülasyonu YARATMAZ, yalnız var olsaydı hangi timing alanlarının kullanılacağını hazırlar). W8/W9 — 8B sonrası mezarlık/OD-008 kontrolünden TEKRAR geçmeli. W12 — W8'e bağımlı, W8 açılmadan açılamaz.

**Kodlamaya başlanmadı.** Operatör Phase 7A'nın tam kapsamını henüz onaylamadı; bunun yerine dar kapsamlı bir **Phase 7A-0** ön-batch talep etti (aşağı) — PENDING_APPROVAL.

### [2026-07-04, EK] PHASE 7A-0 — EXISTING LIFECYCLE INFRASTRUCTURE CHARACTERIZATION AND SAFETY TEST CLOSURE (PENDING_APPROVAL, kod YAZILMADI)

**Sınırlar (operatör talimatı, aynen):** TESTS_ONLY_WHERE_POSSIBLE · NO_SCHEMA_CHANGE · NO_MIGRATION · NO_BACKFILL · NO_NEW_BINDING · NO_OBSERVER_START · NO_OBSERVER_STOP · NO_RUNTIME_CONFIG_CHANGE · NO_ORDER · NO_PERMISSION_CHANGE · NO_EXISTING_BEHAVIOR_CHANGE · NO_NEW_INDEPENDENT_MD.

**Zemin (bu denetimde doğrulandı, kanıt yukarıda):** `forward_pipeline.py`/`lifecycle/engine.py` hiçbir yerde `governor.authorize()` çağırmıyor (yalnız `promote`/`demote`), `os.environ`/API-key erişimi yok, `execution/`/`risk/`/`brain/` import'u yok. Tüm yeni testler tmp_path fixture DB/ledger üzerinde çalışacak — gerçek `data/ami/research.sqlite`/`knowledge.sqlite`'a YAZILMAYACAK (yalnız mode=ro okuma, tek bir characterization testinde).

**Exact test-file listesi (2 yeni dosya, mevcut production kodu değişmez):**

1. **`tests/test_ami_lifecycle_engine_characterization.py`** (`ami/lifecycle/engine.py` hedefli)
   - deterministic signal identity — `classify_lifecycle_path` aynı 1m-path girdisine her zaman aynı state-dizisini üretir (saklı durum yok)
   - known_at_ts / no-lookahead — `_mark_path_1m`'e W10a paterniyle: anchor'dan SONRA eklenen uç bir mark_price satırı, anchor'dan önceki path noktalarını değiştirmez
   - invalid transition sequence rejection (characterization) — dizi hiçbir zaman ardışık-yinelenen state içermez, her zaman OPEN'da başlar CLOSED'da biter (mevcut collapse-mantığının kilidi)
   - lifecycle current-state rebuild from ledger — aynı fixture ledger'ı 2× `replay_shadow_ledger` ile okumak birebir aynı sonucu üretir (saklı/mutasyona uğrayan durum yok)
   - late and out-of-order observations — ledger satırları karıştırılmış sırada okunduğunda per-id sınıflandırma DEĞİŞMEZ (anahtar ledger "id"si, dosya konumu değil)
   - no import of order router/executor/position manager — kaynak dosyanın import listesinin statik testi
   - no trading credential requirement — `os.environ`/API-key erişimi olmadığının statik testi

2. **`tests/test_ami_research_forward_pipeline_characterization.py`** (`ami/research/forward_pipeline.py` hedefli)
   - deterministic signal identity — trade dedup anahtarı (`tid = id or f"{signal}:{entry_ts_ms}"`) formülünün karakterizasyonu
   - restart idempotency — `run_once()` aynı fixture ledger+DB'de 2× çağrılır; 2. çağrıda accepted/rejected=0, duplicates=tümü
   - duplicate transition suppression — aynı trade id iki kez sunulduğunda `processed_trades` PK-öncesi SELECT-check ile engellendiği (DB constraint hatasına düşülmeden) karakterize edilir
   - append-only enforcement — `forward_pipeline.py` kod yolunda `processed_trades`/`forward_bindings`'e hiçbir DELETE/UPDATE olmadığının statik+dinamik testi
   - historical replay does not increase FORWARD_N — fixture ledger'da `entry_ts_ms <= frozen_ms` olan satırlar PRE_FREEZE ile reddedilir, `processed_trades` sayısı **0** kalır
   - activation boundary — `entry_ts_ms == frozen_ms` (tam sınır) → PRE_FREEZE reddi; `== frozen_ms+1` → kabul-edilebilir (exact `<=` sınır testi)
   - historical/proxy/forward separation — bu modülün gerçek ayrım mekanizması R1/PRE_FREEZE'dir (proxy-popülasyon kavramı bu katmanda YOK, o Master-Roadmap `SourceQuality` katmanına ait — kapsam sınırı açıkça not edilecek, uydurulmayacak)
   - crash before commit — bağımsız bağlantıyla INSERT yapılıp commit edilmeden bağlantı kapatılır; yeniden açıldığında satır YOK, trade yeniden işlenebilir durumda
   - crash after commit — commit edilmiş satır, yeniden çalıştırmada doğru şekilde duplicate sayılır, evidence tekrar eklenmez
   - partial-batch recovery (characterization) — commit granularitesinin PER-BINDING olduğu (per-trade DEĞİL) karakterize edilir: simüle edilmiş yarıda-kesilme, o binding'in TÜM batch'ini geri alır (kısmi satır bozulması yok, güvenli tam-yeniden-işleme)
   - late and out-of-order observations — ledger satır sırası karıştırıldığında sonuç değişmez (dedup anahtarı `id`, dosya sırası değil)
   - no import of order router/executor/position manager — statik test
   - no trading credential requirement — statik test
   - **mutation test proving no order creation** — `governor.authorize()` bu modülde HİÇ çağrılmadığının statik kilidi + `_governor_review`'in yalnız `promote`/`demote` çağırdığının ve `PROMOTION_GATES` frozen+evidence şartı sağlanmadan asla `LIVE_ALLOWED`/`SIZING_ALLOWED` gerektiren bir statüye atlayamayacağının (ladder sırası + gate kontrolü) karakterizasyonu
   - **existing two bindings remain behaviorally unchanged** — gerçek `data/ami/research.sqlite`'a **yalnız `mode=ro`** bağlantıyla, `E-HOUR17-FWD-001`/`E-CONVCOMP-FWD-001`'in `spec_hash`/`frozen_ms`/`dataset_hash`/`execution_model`/`candidate_version` alanlarının bu denetimde kaydedilen TAM değerlerle (yukarıdaki tablo) birebir eşleştiğinin kilidi (FORWARD_N gibi doğal-büyüyen bir sayı DEĞİL — bu alanlar rebind olmadıkça sabit kalması gereken kimlik alanları, bu yüzden kalıcı bir regresyon testi olarak güvenli)
   - **protected session delta remains zero** — batch sonunda `git status --short -- tools/s34_state_machine_live_executor.py .env execution/ risk/ brain/ tools/s34_realtime_shadow_runner.py` çıktısının bu batch başlangıcındaki ile BİREBİR aynı olduğunun testi (subprocess ile git status çağrılıp diff'siz olduğu doğrulanır)

**Doğrulama adımı (pytest dışı, manuel):** batch başında ve sonunda `git status --short` (tam repo) + iki protected dosyanın sha256'sı karşılaştırılacak — bu response'ta zaten alınan referans değerler (yukarı) batch-başı taban olarak kullanılacak.

**Kodlamaya başlanmadı.**

WAIT_FOR_OPERATOR_APPROVAL

### [2026-07-04, EK] PHASE 7A-0 TAMAMLANDI — APPROVE PHASE 7A-0 TEST CLOSURE

Operatör onayı ("APPROVE PHASE 7A-0 TEST CLOSURE") ile yalnız 2 test dosyası eklendi; **production kodu SIFIR değişti.**

**Exact changed files:** `tests/test_ami_lifecycle_engine_characterization.py` (YENİ, 11 test), `tests/test_ami_research_forward_pipeline_characterization.py` (YENİ, 17 test). Başka HİÇBİR dosya değişmedi (`ami/lifecycle/engine.py`, `ami/research/forward_pipeline.py`, `ami/governance/governor.py`, şema dosyaları — hepsi dokunulmadı).

**Full test result:** 11/11 + 17/17 = **28/28 ✓** (ayrı ayrı ve birlikte çalıştırıldı). **Toplam AMI: 407/407 ✓** (379+28).

**Operatörün 3. düzeltmesi (activation boundary) — characterization sonucu:**
```
PREFERRED CONTRACT:  event_ts <  activation_ts -> HISTORICAL_REPLAY
                     event_ts >= activation_ts -> FORWARD_OBSERVATION
CURRENT CODE:        entry_ts_ms <= frozen_ms  -> PRE_FREEZE (HISTORICAL)   [forward_pipeline.py run_once()]
                     entry_ts_ms >  frozen_ms  -> forward-eligible
```
**Uyuşmazlık yalnız tam sınırda (`==`):** kod, tam eşitliği HISTORICAL sayıyor; tercih edilen sözleşme FORWARD sayardı. **Sessizce değiştirilmedi** (`test_activation_boundary_characterization` bu davranışı kilitliyor). Bu, tercih edilen sözleşmeden DAHA SIKI bir yön (bir tie hiçbir zaman yanlışlıkla forward sayılmıyor) — bu yüzden çekirdek no-lookahead güvenlik gereksinimi ("event_ts < activation_ts olan hiçbir şey asla forward sayılmaz") ihlal edilmiyor; yalnız tam sınırdaki KURAL TERCİHİ farklı. Ayrı bir karar maddesi olarak aşağıda MET/NOT_MET ayrımında ele alındı.

**Operatörün 8. düzeltmesi (partial-batch recovery) — pre-batch plandaki varsayım DÜZELTİLDİ:** Önceki Phase 7A-0 planı "commit granularitesi per-binding" varsaymıştı. Gerçek karakterizasyon (`test_partial_batch_commit_granularity_characterization` + kod okuması: `ResearchRegistry.attach_evidence()` kendi içinde `commit()` çağırıyor) **daha ince granülerlik** ortaya çıkardı: kabul edilen (accepted) bir trade'in `processed_trades` satırı + `evidence` satırı BİRLİKTE, HEMEN commit edilir (attach_evidence'ın kendi commit'i, aynı bağlantı/transaction). Reddedilen/duplicate trade'lerin `processed_trades` satırı ise bir SONRAKİ accepted trade'in commit'ine VEYA binding-döngüsü sonundaki commit'e kadar commit edilmez. Sonuç aynı (crash-safe, deterministik yeniden-işlenebilir) ama mekanizma daha ince taneli — dürüstçe düzeltildi, production kod değişmedi.

**Safety invariant sonuçları:**
| Invariant | Sonuç |
|---|---|
| Known-at/no-lookahead (`_mark_path_1m`, lifecycle) | **MET** |
| No-lookahead çekirdek garantisi (event_ts<activation_ts asla forward sayılmaz) | **MET** |
| Activation-boundary TAM SINIR sözleşme eşleşmesi | **MISMATCH (characterized, NOT a safety violation)** — ayrı karar maddesi, aşağı |
| Append-only enforcement | **MET** |
| Restart idempotency / duplicate suppression | **MET** |
| Historical replay FORWARD_N'i artırmıyor | **MET** |
| INDEPENDENT_FORWARD_CYCLE_N / FORWARD_EXPERIMENT_PROGRESS / FORWARD_READINESS_N | **NOT_IMPLEMENTED** (bu metrikler mevcut kodda hiç yok — ihlal değil, yokluk) |
| Crash-before-commit / crash-after-commit / partial-batch recovery | **MET** |
| No order-router/executor/position-manager import | **MET** |
| No trading credential requirement | **MET** |
| No order creation (authorize() hiç çağrılmıyor; promote() LIVE/SIZING izni vermiyor) | **MET** |
| 2 mevcut binding'in kimlik alanları (spec_hash/frozen_ms/dataset_hash/candidate_version/signal/knowledge_id) değişmedi | **MET** (mode=ro okuma ile doğrulandı) |
| Protected session delta | **MET — ZERO** |

**Blockers:** YOK. Hiçbir safety invariant NOT_MET çıkmadı (yalnız 1 characterization-level sözleşme-tercihi uyuşmazlığı, güvenlik ihlali değil).

**Protected session delta (batch başı → sonu):**
```
tools/s34_state_machine_live_executor.py sha256: a8067f2c2575e60049382671fa955878b442c59178b28d6ab8cbef44a2804be0  (DEĞİŞMEDİ)
tools/s34_realtime_shadow_runner.py      sha256: a326a5c43b5c4095840ac9fc0f89a22f15f8e4d9aa75c9946c6ea9991d165e27  (DEĞİŞMEDİ)
execution/ risk/ brain/ .env  git status: (boş — 0 satır)
SESSION_DELTA_ON_PROTECTED_FILES = ZERO
```

**Schema unchanged evidence:** `ami/warehouse/schema.py` dokunulmadı, `CANONICAL_SCHEMA_VERSION` hâlâ 7 (kontrol edildi); testler kendi `ResearchRegistry`/`KnowledgeStore` şemalarını yalnız `tmp_path` fixture dosyalarında `executescript` ile kurdu (bu sınıfların KENDİ mevcut `__init__`'i, yeni kod değil).

**Migration not executed evidence:** `MIGRATION_LOG.md`'ye YENİ satır EKLENMEDİ (dosyanın kendi kuralı: "yalnız schema/migration değişikliğinde güncellenir" — bu batch'te hiçbiri olmadığı için dosya dokunulmadı, bu da NO_MIGRATION'ın kanıtı).

**Backfill not executed evidence:** `ami_events`/`ami_cycles`/herhangi bir canonical.sqlite tablosuna hiçbir INSERT/UPDATE çağrısı yapılmadı; tüm testler ya saf fonksiyon çağrısı ya da tmp_path fixture DB'si kullandı; gerçek `data/ami/research.sqlite:processed_trades` hâlâ **0 satır** (doğrulandı).

**Observer/runtime unchanged evidence:** `NO_NEW_BINDING`/`NO_OBSERVER_START`/`NO_OBSERVER_STOP`/`NO_RUNTIME_CONFIG_CHANGE` — `ami/run_forward_pipeline.py` çalıştırılmadı, `pipe.bind()` gerçek DB'ye hiç çağrılmadı (yalnız tmp_path fixture'larında doğrudan SQL insert), gerçek `forward_bindings` tablosu hâlâ **2 satır** (değişmedi).

**Order behavior unchanged evidence:** `test_authorize_is_never_called_by_forward_pipeline` + `test_promote_never_grants_live_or_sizing_permission` + statik import-yasağı testleri — hepsi MET.

**Kodlama SINIRLI kaldı** (yalnız 2 test dosyası, production kodu sıfır değişti). Phase 7A'nın kendisi HÂLÂ ayrı bir onay gerektiriyor — bu batch yalnız Phase 7A-0'ı kapatır.

WAIT_FOR_OPERATOR_APPROVAL

## 63. PHASE 7A.1 — CANONICAL LIFECYCLE SCHEMA FOUNDATION + DISPOSABLE HISTORICAL-SAFE RECONSTRUCTION (2026-07-04, Sonnet 5)

**Onay:** "APPROVE PHASE 7A.1 DISPOSABLE IMPLEMENTATION". **Mod:** IMPLEMENT_SCHEMA_AND_MIGRATION_CODE + DISPOSABLE_DB_ONLY — gerçek canonical.sqlite'a **sıfır** schema/data write. `NOT_APPLIED_TO_CANONICAL_DB`.

### Exact changed files (production kod)
- `ami/lifecycle/canonical_schema.py` (YENİ) — şema (2 tablo + 3 index), identity algoritması, transition validator/insert/rebuild, rollback.
- `ami/lifecycle/canonical_backfill.py` (YENİ) — disposable backfill (`ami_events`/`event_cycle_membership` → `ami_signal_lifecycle`/`ami_lifecycle_transitions`), FIELD_CLASSIFICATION matrisi.
- `ami/lifecycle/migration_rehearsal.py` (YENİ) — 14-adım disposable rehearsal orkestratörü.
- **`ami/warehouse/schema.py` DOKUNULMADI** (bilinçli tasarım kararı: yeni tablolar paylaşılan `_SCHEMA`/`init_schema()`'ya eklenirse, `DEFAULT_PATH`'e karşı çalışan HERHANGİ bir mevcut script (`main()` fonksiyonları, cron) sonraki çalıştırmasında YANLIŞLIKLA gerçek DB'ye uygulardı — bu riski tamamen ortadan kaldırmak için tamamen AYRI bir şema modülü tercih edildi).

### Exact test-file changes
- `tests/test_ami_lifecycle_canonical_schema.py` (YENİ, 32 test)
- `tests/test_ami_lifecycle_canonical_backfill.py` (YENİ, 11 test)
- `tests/test_ami_lifecycle_migration_rehearsal.py` (YENİ, 4 test — gerçek canonical.sqlite'ın disposable KOPYASINA karşı)

### Schema version
- **Before:** `CANONICAL_SCHEMA_VERSION = 7` (değişmedi, dokunulmadı)
- **Proposed after** (yalnız gelecekteki ayrı "APPROVE PHASE 7A CANONICAL MIGRATION" onayıyla): 8

### Exact schema objects added (disposable modülde, canonical.sqlite'a UYGULANMADI)
```sql
CREATE TABLE ami_signal_lifecycle (
  signal_id TEXT PRIMARY KEY, setup_id, setup_version, source_event_id, independent_cycle_id,
  symbol, direction, timeframe, route_version, signal_birth_ts, first_known_ts, first_executable_ts,
  last_confirmation_ts, invalidation_ts, terminal_ts, lifecycle_status, lifecycle_reason_code,
  observation_mode, evidence_layer, is_proxy, executability_status, identity_version, schema_version,
  source_hash, code_commit, provenance, created_at, updated_ms
  + 7 CHECK constraint (direction/evidence_layer-is_proxy tutarlılığı, timestamp sıralaması)
)
CREATE TABLE ami_lifecycle_transitions (
  transition_id TEXT PRIMARY KEY, signal_id, previous_status, new_status, transition_ts, known_at_ts,
  reason_code, transition_version, observation_mode, correction_of, evidence_ref, schema_version,
  provenance, created_ms
  + UNIQUE(signal_id,previous_status,new_status,transition_ts,reason_code,transition_version,observation_mode)
  + CHECK(known_at_ts>=transition_ts)
)
+ 3 index (signal_id+transition_ts, source_event_id, independent_cycle_id)
```

### Identity algoritması
`generate_signal_id(setup_id, setup_version, symbol, direction, source_event_id|signal_birth_ts)` — `ami.identity.event_identity.generate_event_id`'in birebir aynı deterministik-hash-doğal-anahtar paterni (SHA256[:24], `SIG-` prefix). Event-anchored: `EVENT|{source_event_id}`. Event-less: `NOEVENT|{signal_birth_ts}` (signal_birth_ts zorunlu, yoksa `ValueError` — sahte anchor icat edilmez).

**Tasarım düzeltmesi (önemli bulgu):** Erken Phase 7-8 audit'in "signal_id=event_id 1:1" basitleştirmesi bu batch'te TERK EDİLDİ. Gerçek veri incelemesi (`route_version` sütunu) gösterdi ki bazı anchor'lara **hem LONG_\* hem SHORT_\* route'lar aynı anda** bağlı (örn. `"LONG_ECHO_45_120_SILENCE,LONG_OFI_SILENCE_BUYERS,LONG_T15_BOUNCE,SHORT_NOISY_BTC1M_D5_H180"`) — tek bir event için tek bir `direction` alanı zorunluluğuyla (operatörün identity contract'ı) çelişiyordu. Çözüm: **signal = (source_event_id, setup_token) çifti** — bir anchor'a N route bağlıysa N ayrı signal_id üretiliyor, her biri kendi `direction`ıyla. Gerçek veri: 252 event → **270 signal** (18 event'in 2. bir route'u var).

### Migration sequence (14 adımdan 12'si bu batch'te; 13-14 ayrı raporlandı)
1. Schema fingerprint (gerçek DB, mode=ro) → 2. Disposable copy (`shutil.copy2`) → 3. Additive migration → 4. Constraints/index (şemanın kendisi) → 5. Backfill (1. çalıştırma) → 6. Backfill (2. çalıştırma) → 7. Row-count karşılaştırma → 8. Content-hash karşılaştırma → 9. Current-state rebuild → 10. Rollback rehearsal → 11. Migration yeniden uygulama → 12. Old-reader compatibility → **(13. full tests ve 14. protected-diff gate bu modülün DIŞINDA, ayrı, portable-olmayan kontroller olarak çalıştırıldı — Phase 7A-0'ın "git status'u pytest içine hatalı gömme" dersi burada da uygulandı)**.

### Disposable DB source/copy method
`shutil.copy2(data/ami/canonical.sqlite, tmp_path/disposable.sqlite)` — kaynağın yalnız OKUNMASI (dosya sistemi read), hedefe YAZMA. Gerçek dosya hiçbir zaman yazılabilir modda açılmadı (yalnız `mode=ro` fingerprint/count için).

### Disposable migration result (gerçek veri, disposable kopya üzerinde)
| Kontrol | Sonuç |
|---|---|
| `schema_fingerprint_before` ≠ `schema_fingerprint_after_migration` | ✅ (migration gerçekten çalıştı) |
| `new_tables_present` | ✅ True |
| Deterministic backfill (1. çalıştırma) | **signals_upserted=270, transitions_attempted=536, source_event_n=252** |
| Backfill 2. çalıştırma (rerun) | **270/536/252 — BİREBİR AYNI** |
| Row-count equality (1↔2) | ✅ True |
| Content-hash equality (1↔2) | ✅ True |
| Current-state rebuild equality | ✅ True (270/270 signal, 0 mismatch) |
| Rollback rehearsal (tablolar kaldırıldı) | ✅ True |
| Rollback sonrası mevcut tablolar korundu (`ami_events` count değişmedi) | ✅ True |
| Migration yeniden uygulama (reapply) sonrası sayılar | **270/536 — pre-rollback ile BİREBİR AYNI** |
| Old-reader compatibility (`ami_events` count + `feature_gateway.fetch_events`/`fetch_chart_feature`) | ✅ True (değişmedi) |

### Field-classification matrix (backfill, `ami/lifecycle/canonical_backfill.py:FIELD_CLASSIFICATION`)
| Alan | Sınıf | Gerekçe |
|---|---|---|
| source_event_id, independent_cycle_id, signal_birth_ts, terminal_ts, symbol, setup_id, route_version, evidence_layer, is_proxy | **DETERMINISTIC_HISTORICAL_SAFE** | Doğrudan `ami_events`/`event_cycle_membership`'ten, değişmeden |
| direction | **HISTORICAL_PROXY** | `setup_id` route-adı önek-ayrıştırması (LONG_/SHORT_/BUY_FADE) — canonical route-registry yok, tanınmayan önek → UNKNOWN (uydurulmadı) |
| setup_version | **NOT_IMPLEMENTED** | Route-bazlı kural/parametre versiyonlama hiçbir yerde yok; dondurulmuş yer-tutucu sabit (`setup-v1`) kullanıldı |
| timeframe, first_known_ts, last_confirmation_ts, invalidation_ts | **NOT_IMPLEMENTED** | `feature_available_ts_ms` hep NULL (üretilmemiş); "confirmation"/"invalidation" path-metrics motoru gerektiriyor (bu batch'te yasak: NO_TIMING_PATH_LABEL_ENGINE) |
| first_executable_ts | **FORWARD_ONLY** | Gerçek executability zamanı candle/mark proxy'sinden geriye dönük üretilemez (operatör talimatı) — NULL bırakıldı |

Hiçbir alan zero'ya çevrilmedi; hepsi NULL + açık sınıflandırma.

### Second-run idempotency / row-count / content-hash / current-state rebuild / rollback / reapply / old-reader — hepsi yukarıdaki tabloda ✅

### Full test result
`test_ami_lifecycle_canonical_schema.py` 32/32 ✓ + `test_ami_lifecycle_canonical_backfill.py` 11/11 ✓ + `test_ami_lifecycle_migration_rehearsal.py` 4/4 ✓ = **47/47 yeni ✓**. Regresyon kontrolü (Phase 7A-0 dosyaları + registry_seed) ayrıca çalıştırıldı, hepsi yeşil. **Toplam AMI: 454/454 ✓** (407+47).

### Protected hashes before/after
```
tools/s34_state_machine_live_executor.py sha256: a8067f2c2575e60049382671fa955878b442c59178b28d6ab8cbef44a2804be0  (DEĞİŞMEDİ)
tools/s34_realtime_shadow_runner.py      sha256: a326a5c43b5c4095840ac9fc0f89a22f15f8e4d9aa75c9946c6ea9991d165e27  (DEĞİŞMEDİ)
execution/ risk/ brain/ .env  git status: (boş — 0 satır)
```
**SESSION_DELTA_ON_PROTECTED_FILES = ZERO.**

### Unresolved blockers
**YOK.** Hiçbir DESIGN_BLOCKER/MIGRATION_BLOCKER/SAFETY_BLOCKER/COMPATIBILITY_BLOCKER oluşmadı. Tasarım düzeltmesi (signal_id=event_id basitleştirmesinin terk edilmesi) bir blocker değil, bir iyileştirme olarak ele alındı ve şeffaf raporlandı.

### Phase 7A canonical-migration readiness verdict
**HAZIR (disposable kopya üzerinde tam doğrulandı)** — ancak gerçek DB'ye uygulama, operatörün açıkça belirttiği gibi, YALNIZ ayrı bir "APPROVE PHASE 7A CANONICAL MIGRATION" onayıyla yapılabilir; bu batch bunu YAPMADI.

```
Implementation code prepared: YES
Disposable migration executed: YES
Canonical DB migration executed: NO
Canonical DB backfill executed: NO
Observer started: NO
Existing binding changed: NO
Order/live/shadow behavior changed: NO
```

**Migration log status (M-0022):** `PREPARED_AND_VALIDATED_ON_DISPOSABLE_COPY` / `NOT_APPLIED_TO_CANONICAL_DB`.

WAIT_FOR_OPERATOR_APPROVAL

## 64. PHASE 7A CANONICAL MIGRATION — APPLIED TO REAL canonical.sqlite (2026-07-04, Sonnet 5)

**Onay:** "APPROVE PHASE 7A CANONICAL MIGRATION". **Mod:** CANONICAL_DB_MIGRATION_ALLOWED + CANONICAL_HISTORICAL_SAFE_BACKFILL_ALLOWED. NO_PHASE_7B/NO_TIMING_PATH_ENGINE/NO_FORWARD_BINDING_CHANGE/NO_OBSERVER_START-STOP/NO_RUNTIME_CONFIG_CHANGE/NO_ORDER/NO_PERMISSION_CHANGE/NO_LIVE_SHADOW_EXECUTOR_CHANGE/NO_DASHBOARD_EXTENSION — hepsi korundu.

### Canonical DB path
`D:\eclipse_scalper\data\ami\canonical.sqlite`

### Backup
`data/ami/backups/canonical_pre_phase7a_migration_20260704_130548.sqlite` — sha256=`d5aa43313ae82241961587b237ec24f455728ef34106890bdbddec4dee838dcb` (kaynakla BİREBİR aynı, migration öncesi tam kopya). DB boyutu=178,536,448 byte. WAL/SHM: migration öncesi `wal_checkpoint(TRUNCATE)` ile temizlendi (0 bekleyen sayfa, dosyalar kaldırıldı) — backup tek-dosya tutarlı. Serbest disk: D:\ 1.1TB (yeterli). Git commit: `5cda3122cf507439779092c4453674e8f02d9e8d` (bu oturumda commit yapılmadı). Migration-code hash (değişmeden kullanıldı): `canonical_schema.py`=`0e5a52d2...`, `canonical_backfill.py`=`c6e273b2...`, `migration_rehearsal.py`=`da2f7669...` (Phase 7A.1'de doğrulanan HAŞLARLA birebir aynı).

### Preflight sonuçları
- **Schema fingerprint uyumu:** gerçek DB'nin migration-öncesi fingerprint'i (`dca3fba1...`) Phase 7A.1'in disposable rehearsal'ında kullanılan kaynağın `schema_fingerprint_before` değeriyle **birebir eşleşti** — SCHEMA_DRIFT_BLOCKER tetiklenmedi.
- **Concurrent writer kontrolü:** `BEGIN IMMEDIATE` anında kilit alındı (aktif yazıcı yok) — ACTIVE_WRITER_BLOCKER tetiklenmedi. Hiçbir proses durdurulmadı/config değiştirilmedi.
- **Frozen source snapshot:** migration öncesi `ami_events`=252 satır okundu; `derive_signals()` ile deterministik beklenen sayı **270 signal / 536 transition** hesaplandı (disposable referansla birebir — popülasyon değişmemiş, kör zorlama gerekmedi).

### Exact migration applied
`ami/warehouse/schema.py`: `CANONICAL_SCHEMA_VERSION` **7→8**; yeni `_SCHEMA_PHASE7A` bloğu eklendi (Phase 7A.1'de doğrulanan SQL'in **birebir kopyası** — `ami/lifecycle/canonical_schema.py`'deki `_SCHEMA` değiştirilmedi, iki yerde aynı DDL var, drift riski docstring'de not edildi); `init_schema()` artık bu bloğu da çalıştırıyor. **`ami/lifecycle/{canonical_schema,canonical_backfill,migration_rehearsal}.py` SIFIR değişti** (hash-doğrulandı, migration/backfill mantığı Phase 7A.1'de doğrulananla birebir aynı kod).

### Backfilled counts (gerçek DB)
signals_upserted=**270**, transitions_attempted=**536**, source_event_n=**252** — disposable referansla birebir aynı.

### Field-classification counts (gerçek veri)
evidence_layer: REAL=270/PROXY=0. observation_mode: HISTORICAL_REPLAY=270. lifecycle_status: CLOSED=266/OPEN=4 (henüz terminal_ts'i olmayan 4 anchor — RIGHT_CENSORED benzeri, uydurulmadı). first_executable_ts NOT NULL olan satır=**0** (FORWARD_ONLY korunuyor).

### Current-state rebuild result
270/270 signal için `rebuild_current_state()` ledger'dan yeniden inşa edilen durum, denormalize `lifecycle_status` kolonuyla **birebir eşleşti** — 0 mismatch.

### Second-run idempotency result
Backfill 2. kez çalıştırıldı: signals_upserted=270/transitions_attempted=536 (aynı) — row-count DEĞİŞMEDİ, content-hash DEĞİŞMEDİ, 0 yeni duplicate.

### Content hashes
Run-1 ile run-2 arası `content_hash_lifecycle_tables()` **birebir aynı** (exact hash değeri raporlanmadı, yalnız eşitlik — hash'in kendisi disposable rehearsal testlerinde zaten sabitlenmiş).

### FORWARD_N before/after
`data/ami/research.sqlite:processed_trades` = **0 → 0** (değişmedi; bu migration hiçbir forward-evidence akışına dokunmadı).

### Existing binding before/after
İki binding de (`E-HOUR17-FWD-001`, `E-CONVCOMP-FWD-001`) `spec_hash`/`frozen_ms`/`dataset_hash`/`candidate_version`/`signal`/`knowledge_id` alanlarında **BİREBİR AYNI** (Phase 7A-0'da kaydedilen referans değerlerle eşleşti).

### Old-reader compatibility
`feature_gateway.fetch_events` (252), `fetch_cycles` (167), `fetch_chart_feature:ami_candles` (173464) — migrasyon sonrası hepsi **beklenen sayılarla birebir eşleşti**.

### Full test result
38 `test_ami_*.py` dosyasının TAMAMI + `test_buyfade_mutations.py`/`test_buyfade_silexit_mutations.py` migrasyon SONRASI gerçek DB'ye karşı tek tek (2'şer dosya) yeniden çalıştırıldı — **hepsi ✓**. **1 characterization bulgusu:** `test_schema_fingerprint_changes_only_by_addition` migrasyon-öncesi bir varsayımla yazılmıştı ("disposable kopya migration'ı görünür şekilde değiştirir") — migration artık GERÇEKTEN uygulandığı için kaynak zaten tabloları içeriyor, disposable-kopya migration'ı bu durumda no-op (fingerprint değişmiyor). Bu bir **regresyon/veri-bütünlüğü sorunu DEĞİL** — migration'ın başarıyla kalıcı olduğunun bağımsız kanıtı; test her iki duruma (migrasyon-öncesi/sonrası kaynak) karşı sağlam olacak şekilde düzeltildi (`new_tables_present` her zaman kontrol edilir; fingerprint-farkı yalnız kaynak henüz migrasyonsuzsa zorunlu tutulur). **Toplam AMI: 454/454 ✓** (test sayısı değişmedi, yalnız 1 assertion post-migration gerçeğini yansıtacak şekilde düzeltildi).

### Protected hashes before/after
```
tools/s34_state_machine_live_executor.py sha256: a8067f2c2575e60049382671fa955878b442c59178b28d6ab8cbef44a2804be0  (DEĞİŞMEDİ)
tools/s34_realtime_shadow_runner.py      sha256: a326a5c43b5c4095840ac9fc0f89a22f15f8e4d9aa75c9946c6ea9991d165e27  (DEĞİŞMEDİ)
execution/ risk/ brain/ .env  git status: (boş — 0 satır)
```
**SESSION_DELTA_ON_PROTECTED_FILES = ZERO.**

### Rollback/restore readiness
Backup dosyası mevcut ve hash-doğrulanmış (`data/ami/backups/canonical_pre_phase7a_migration_20260704_130548.sqlite`). Acil durumda restore = backup dosyasını `data/ami/canonical.sqlite` üzerine kopyala (additive migration olduğu için normalde gerekmez — Phase 7A.1'in disposable rehearsal'ı zaten rollback'in `DROP TABLE IF EXISTS` ile temiz çalıştığını kanıtladı).

### Unresolved blockers
**YOK.** 0 DESIGN_BLOCKER/MIGRATION_BLOCKER/SAFETY_BLOCKER/COMPATIBILITY_BLOCKER/DATA_INTEGRITY_BLOCKER.

### Phase 7B readiness verdict
Phase 7A canonical migration TAMAMLANDI ve tam doğrulandı. **Phase 7B'ye OTOMATİK GEÇİLMEDİ** — operatörün "Bu onay Phase 7B veya Phase 8 onayı değildir" talimatı gereği ayrı bir onay bekleniyor.

```
Canonical DB migration executed: YES
Canonical DB backfill executed: YES
Observer started: NO
Existing binding changed: NO
FORWARD_N increased: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

### [2026-07-04, EK — operatör provenance tutarlılık kontrolü] PHASE_7A_PROVENANCE_GAP

Operatör §64'teki "REAL=270/PROXY=0" (evidence_layer) ile Phase 7A.1 raporundaki "direction=HISTORICAL_PROXY" (field classification) arasındaki tutarlılığı sorguladı. Read-only SQL doğrulaması (kod/migration DOKUNULMADI):

```sql
SELECT evidence_layer, is_proxy, COUNT(*) FROM ami_signal_lifecycle GROUP BY evidence_layer, is_proxy;
-- [('REAL', 0, 270)]   -- ami_events.source_quality'den (100% REAL_LIQUIDATION), direction'la ilgisiz
PRAGMA table_info(ami_signal_lifecycle);
-- provenance kolonu var ama field-level classification/method kolonu YOK
SELECT DISTINCT provenance FROM ami_signal_lifecycle;
-- [('batch-p7a-canonical-migration',)]  -- 270 satırda AYNI, batch-etiketi; per-field değil
```

**Bulgu:** `FIELD_CLASSIFICATION["direction"]="HISTORICAL_PROXY"` yalnız `ami/lifecycle/canonical_backfill.py`'de bir Python sözlüğü — canonical.sqlite'ta HİÇBİR tabloya/kolona/metadata satırına yazılmadı. `direction` kolonu düz `LONG`/`SHORT` değerleriyle saklanıyor, hiçbir flag/companion-kolon onun route-adı önek-ayrıştırmasıyla (heuristik, ground-truth değil) türetildiğini işaretlemiyor. Nesir raporlarda (§63/§64) dürüstçe belirtildi ama **veri/şema seviyesinde bu caveat kayıp** — projenin `source_quality`/`evidence_layer`/`identity_version` gibi alanlarda zaten uyguladığı "provenance şema-seviyesinde birinci-sınıf" disipliniyle tutarsız.

**Sınıflandırma: `PHASE_7A_PROVENANCE_GAP`.** Başarı olarak sunulmuyor. **Phase 7B'ye geçilmedi.**

WAIT_FOR_OPERATOR_APPROVAL

## 65. PHASE 7A-P1 — DISPOSABLE FIELD-LEVEL PROVENANCE CLOSURE (2026-07-04, Sonnet 5)

**Onay:** "APPROVE PHASE 7A-P1 DISPOSABLE PROVENANCE CLOSURE". **Mod:** DISPOSABLE_DB_ONLY/NO_CANONICAL_DB_WRITE — gerçek canonical.sqlite'a sıfır write (doğrulandı: provenance-ilişkili hiçbir obje yok, ami_signal_lifecycle/ami_lifecycle_transitions=270/536 değişmedi).

### Reconciliation (yeni tasarımdan ÖNCE)
Mevcut 5 tablo incelendi — hiçbiri per-field provenance karşılamıyor: `data_quality_events` (feed-seviyesi gap/stale event), `causal_assumption_registry` (causal-DAG confounder), `evidence_contamination` (hipotez kontaminasyonu), `mt_family_registry` (multiple-testing istatistiği), `market_structure_versions` (fee/contract versiyonlama). `ami_signal_lifecycle.provenance` yalnız düz bir batch-etiketi (270 satırda AYNI değer) — per-field değil. **existing provenance equivalent found: NO.**

### Exact schema object proposed
`ami_lifecycle_field_provenance` (genel sözleşme, direction'a özel değil — signal_id/field_name/field_classification/is_proxy/derivation_method/source_reference/limitations/provenance_version/schema_version/code_commit/source_hash/created_at) + `ami_lifecycle_direction_view` (canonical query contract — direction hiçbir zaman classification'sız sunulmaz). `UNIQUE(signal_id,field_name,provenance_version)` + `CHECK(field_classification IN (5 değer))` + `CHECK(is_proxy tutarlılığı)` + `FOREIGN KEY(signal_id)→ami_signal_lifecycle`.

### Direction derivation (gerçek kod incelendi, tahmin edilmedi)
`ami.lifecycle.canonical_schema.classify_direction_from_setup_id()` — `setup_id` önek eşleşmesi: `LONG_`→LONG, `SHORT_`/`BUY_FADE`→SHORT (failure_archive #8/#19 emsaliyle), aksi→UNKNOWN. `test_direction_provenance_matches_actual_derivation_code` bunu canlı kod çağrısıyla çapraz-doğruladı (hardcoded varsayım değil).

### Disposable rehearsal sonucu (gerçek DB'nin disposable kopyası, schema-v8 üzerine)
| Kontrol | Sonuç |
|---|---|
| Provenance satırları (270 signal × 16 alan) | **4320** |
| 2. çalıştırma (duplicate kontrolü) | 0 yeni satır, row-count+content-hash eşit |
| `direction` dağılımı | **270/270 = HISTORICAL_PROXY / is_proxy=1** |
| Missing-provenance validator | Eksik provenance'ta GERÇEKTEN reddediyor (`FieldProvenanceViolation`) — sessizce REAL varsaymıyor |
| 270 signal / 536 transition | **DEĞİŞMEDİ** |
| FORWARD_N | 0 → 0 (değişmedi) |
| Rollback | Yeni obje temiz kaldırıldı, lifecycle tabloları korundu |
| Reapply | Rollback-öncesi sayılarla birebir |
| Old-reader (`feature_gateway.fetch_events`) | 252 — etkilenmedi |

### Semantik netlik (row-level vs field-level, `SCHEMA_DICTIONARY.md`'ye işlendi)
`ami_signal_lifecycle.is_proxy=0` (REAL anchor) ile `ami_lifecycle_field_provenance`'taki `direction` alanının field-level `is_proxy=1` (HISTORICAL_PROXY) olması **eş zamanlı ve tutarlı** — iki farklı eksen. `ami_lifecycle_direction_view` bu ayrımı canonical query-seviyesinde zorluyor (ham `direction` sorgusu hâlâ mümkün ama yalnız iç-kullanım, dışa sunum için view kullanılır).

### Full test result
15 (`test_ami_lifecycle_canonical_field_provenance.py`) + 3 (`test_ami_lifecycle_provenance_rehearsal.py`) = **18 yeni ✓**. 38 `test_ami_*.py` + buyfade/silexit dosyaları TAM regresyon için tek tek yeniden çalıştırıldı, hepsi ✓. **Toplam AMI: 472/472 ✓** (454+18).

### Protected hashes
Değişmedi (hash'ler §63/§64'teki referans değerlerle birebir aynı). **SESSION_DELTA_ON_PROTECTED_FILES = ZERO.**

### Unresolved blockers
YOK.

### Canonical provenance migration readiness verdict
**HAZIR** (disposable kopyada tam doğrulandı) — gerçek DB'ye uygulama yalnız ayrı "APPROVE PHASE 7A-P CANONICAL PROVENANCE MIGRATION" onayıyla yapılabilir.

```
Disposable provenance migration executed: YES
Canonical DB migration executed: NO
Canonical lifecycle rows modified: NO
Observer/binding changed: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

## 66. PHASE 7A-P1 — SCOPE AND VERSION CORRECTION (v9, 16-FIELD) WITH SEMANTIC CLOSURE (2026-07-04, Sonnet 5)

**Onay:** "APPROVE PHASE 7A-P1 SCOPE AND VERSION CORRECTION (v9, 16-FIELD) — WITH SEMANTIC CLOSURE". 16-field provenance kapsamı kabul edildi, canonical `CANONICAL_SCHEMA_VERSION` **v8→v9 gerekli** olarak teyit edildi (bir önceki §65'te kullanılan "8→8" sözleşmesi hatalıydı — v8'in fingerprint'i değişmeden field-provenance objeleri eklemek repo'nun "her şema-şekli değişikliği = version bump" hassasiyetini ihlal ederdi). **Bu batch YİNE disposable-only** — v9 yalnız `ami/lifecycle/canonical_schema.py:LIFECYCLE_SCHEMA_VERSION` (1→2) üzerinde iz bırakıyor, `ami/warehouse/schema.py:CANONICAL_SCHEMA_VERSION` HÂLÂ 8 (gerçek DB'ye v9 bump'ı ayrı bir canonical-migration onayı gerektirir).

### İki semantik düzeltme (disposable copy üzerinde)

1. **`setup_version`**: canonical kolon değeri artık **NULL** (önceden `SETUP_VERSION_DEFAULT`="setup-v1" donmuş sabiti idi). `field_classification=NOT_IMPLEMENTED`, `is_proxy=false`, `derivation_method=not_computed`. `identity_version` ile ASLA karıştırılmıyor — hash-girdisi olarak `SETUP_VERSION_DEFAULT` HÂLÂ kullanılıyor (`generate_signal_id()` değişmedi, `signal_id` sabit kalıyor) ama saklanan kolon değeri artık ayrı ve NULL. Kolon `TEXT NOT NULL` idi → SQLite'ta `ALTER COLUMN` yok, gerçek bir table-rebuild migration'ı (`migrate_setup_version_nullable()`, 12 adım: yeni tablo/kopyala/satır-sayısı-doğrula/eski-sil/yeniden-adlandır/index-yeniden-oluştur/`PRAGMA foreign_key_check`) yazıldı, idempotent (zaten nullable ise no-op, `PRAGMA table_info` ile kontrol).
2. **`terminal_ts`**: canonical kolon değeri artık **NULL** (`ami_events.event_end_ts_ms`'in gerçek lifecycle TERMINAL geçiş zamanı olduğu doğrulanmadı — yalnız source-event'in kendi bitiş zamanı). `field_classification=NOT_IMPLEMENTED`. `event_end_ts_ms` KAYBOLMADI — yeni internal-only `_terminal_transition_ts` alanı ledger-transition kararlarını (536 transition, TERMINAL_CLOSE zamanı) AYNEN eskisi gibi sürdürüyor; yalnız denormalize `ami_signal_lifecycle.terminal_ts` kolonu artık NULL.

### Actual value null/non-null matrisi (provenance-row varlığı DEĞİL, gerçek kolon değeri — 270 signal)

| Alan | null | non_null |
|---|---|---|
| source_event_id, independent_cycle_id, signal_birth_ts, symbol, setup_id, route_version, evidence_layer, is_proxy, direction | 0 | **270** |
| **setup_version** | **270** | 0 |
| **terminal_ts** | **270** | 0 |
| timeframe, first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts | 270 | 0 |

`setup_version_all_null=True`, `terminal_ts_all_null=True` — her ikisi de doğrulandı.

### Değişmeyenler (doğrulandı)
`identity_unchanged_by_semantic_correction=True`, `transitions_unchanged_by_semantic_correction=True` (signal_id/source_event_id/independent_cycle_id kümesi ve 536 transition satırı düzeltme ÖNCESİ/SONRASI birebir aynı küme). `lifecycle_counts_unchanged=True` (270/536). `direction_all_historical_proxy=True` (270/270 HISTORICAL_PROXY/is_proxy=1). `first_executable_ts` FORWARD_ONLY/NULL sabit. FORWARD_N=0 sabit (dokunulmadı).

### Regression bug bulundu ve düzeltildi (bu batch içinde)
`ami/lifecycle/canonical_backfill.py`'nin artık her zaman `setup_version=None` üretmesi, ESKİ (Phase-7A-canonical-migration-track) rehearsal harness'ini (`ami/lifecycle/migration_rehearsal.py:run_disposable_rehearsal()`) kırdı: gerçek canonical.sqlite'ın disposable kopyası ZATEN uygulanmış v8 `setup_version TEXT NOT NULL` kısıtını miras alıyor, `init_lifecycle_schema()`'nın `CREATE TABLE IF NOT EXISTS`'i bu durumda no-op — yeni `migrate_setup_version_nullable()` adımı olmadan `backfill_lifecycle()` `IntegrityError` fırlatıyordu (4/15 test kırmızı). **Düzeltme:** `run_disposable_rehearsal()`'a (2 çağrı noktası — ilk migration + rollback-sonrası-reapply) ve `tests/test_ami_lifecycle_migration_rehearsal.py:test_old_reader_compatibility_feature_gateway_still_works`'ün doğrudan çağrısına `migrate_setup_version_nullable()` eklendi (`provenance_rehearsal.py`'deki desenin birebir aynısı). Re-test sonrası 15/15 ✓.

### Full regression
472/472 mevcut AMI test (37 `test_ami_*.py` + `test_buyfade_mutations.py` + `test_buyfade_silexit_mutations.py`, çift-çift sıralı çalıştırıldı) — **hepsi ✓, toplam AMI 472/472 sabit** (yeni test dosyası eklenmedi; mevcut testler/harness production-kod düzeltmesiyle uyumlu hâle getirildi).

### Rollback/idempotency (disposable, tekrar doğrulandı)
`rollback_removed_new_objects=True`, `rollback_preserved_lifecycle_tables=True`, `reapply_counts_match_pre_rollback=True`, `row_count_equal_across_reruns=True`, `content_hash_equal_across_reruns=True` (`backfill_run1`==`backfill_run2`, 4320/4320 provenance satırı). `old_reader_fetch_events_count=252` (etkilenmedi).

### Content hash'ler (disposable, bu batch'in son çalıştırması)
`lifecycle_tables_content_hash=991f4f12d2083dede5db29c82e10b6d931a22b476ee339d6ea3f34c383f12b6b`, `provenance_table_content_hash=833cd77b90da34f9b66778d7f945325717ebe1eb32160b259af30bda7a8a2791`. Gerçek `data/ami/canonical.sqlite` sha256/mtime bu batch boyunca **DEĞİŞMEDİ** (yalnız disposable kopyalar üzerinde çalışıldı).

### Protected hashes
Bu oturumda `tools/s34_state_machine_live_executor.py`/`execution/`/`risk/`/`brain/`/`.env`/`tools/s34_realtime_shadow_runner.py`'ye HİÇBİR Edit/Write çağrısı yapılmadı — **SESSION_DELTA_ON_PROTECTED_FILES = ZERO** (repo'daki mevcut `git status` farkları bu oturumdan ÖNCEKİ, session-dışı değişikliklerdir).

### Canonical migration readiness verdict
**HAZIR** (disposable kopyada v9/16-field/setup_version-NULL/terminal_ts-NULL semantik kapanışı tam doğrulandı, 472/472 regresyon yeşil) — gerçek DB'ye (`ami/warehouse/schema.py:CANONICAL_SCHEMA_VERSION` 8→9 + `ami_lifecycle_field_provenance`/`ami_lifecycle_direction_view` + `setup_version`/`terminal_ts` NULL-düzeltmesi) uygulama yalnız ayrı, açık bir "APPROVE PHASE 7A-P CANONICAL PROVENANCE MIGRATION" (veya benzeri) onayıyla yapılabilir.

```
Disposable semantic-closure migration executed: YES
Canonical DB migration executed: NO
Canonical lifecycle/provenance rows modified: NO
Observer/binding changed: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

## 67. PHASE 7A-P1 — SEMANTIC CONSISTENCY CHECK ROUND 2: TERMINAL_CLOSE / LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER (2026-07-04, Sonnet 5)

**Onay:** operatörün son read-only-first talimatı — §66'nın terminal_ts=NULL/NOT_IMPLEMENTED düzeltmesiyle, ledger'ın hâlâ `event_end_ts_ms`'ten bir TERMINAL_CLOSE/CLOSED iddiası üretmesi arasındaki çelişkiyi doğrula, gerekiyorsa yalnız disposable copy'de düzelt. **CANONICAL DB'YE DOKUNULMADI. NO PHASE 7B. NO RUNTIME/BINDING/OBSERVER CHANGE** (tam uyulu).

### 1. TERMINAL_CLOSE exact semantics (düzeltme ÖNCESİ karakterizasyon)
`ami_lifecycle_transitions`'a yazılan (artık kaldırılan) transition: `previous_status=OPEN`, `new_status=CLOSED`, `reason_code=TERMINAL_CLOSE`, `transition_ts=known_at_ts=ami_events.event_end_ts_ms` (verbatim). `rebuild_current_state()` etkisi: bu, (transition_ts,transition_version) sıralamasında son satır olduğu için ilgili sinyalin CURRENT durumunu CLOSED yapıyordu (266/270 sinyal).

### 2. Bu transition gerçek SIGNAL LIFECYCLE terminalini mi temsil ediyordu?
**Hayır — yalnız SOURCE EVENT'in gözlem penceresi sonunu (`event_end_ts_ms` = son ilişkili trade kapanışı) temsil ediyordu.** Sinyalin (route-koşullu trading kurgusunun) gerçekten ne zaman kapandığına dair bağımsız hiçbir kanıt (stop/TP/hold-kural/karar-motoru çıktısı) yoktu — §66'da zaten terminal_ts alanı için doğrulanan aynı gerçek, ama ledger/status seviyesinde henüz uygulanmamıştı.

### 3. Taxonomy reconciliation (yeni isim tahmin edilmedi)
`LifecycleReasonCode` (SIGNAL_BIRTH/TERMINAL_CLOSE/DATA_GAP/CORRECTION/HISTORICAL_RECONSTRUCTION) ve `TradeLifecycleState` (bu batch'te yalnız OPEN/CLOSED kullanılabilir — HEALTHY/ACCELERATING/... path-label motoru bu fazda yasak) incelendi: mevcut taksonomide "non-terminal/censored, source-event-ended" karşılığı bir çift **YOK**. Onaylanmamış bir isim (`SOURCE_EVENT_ENDED`/`OBSERVATION_WINDOW_ENDED`) icat ETMEK yerine, en dürüst/aşırı-iddiasız çözüm: bu backfill artık **HİÇBİR** lifecycle_status/transition iddiasını `event_end_ts_ms`'ten türetmiyor — taze backfill edilen her sinyal yalnız OPEN/SIGNAL_BIRTH (genesis). `event_end_ts_ms` kaybolmadı — `ami_events` tablosunda (source-event semantiğiyle, değişmeden) erişilebilir kalıyor, yalnız artık signal-lifecycle terminaliyle karıştırılmıyor.

### 4. LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER
**BULUNDU** (disposable kopyanın başlangıç durumu — gerçek DB'nin M-0023'te uygulanmış hâlinin birebir kopyası): `lifecycle_status` dağılımı CLOSED=266/OPEN=4, ledger'daki `TERMINAL_CLOSE→CLOSED`=266 satırıyla **tutarlı ama YANLIŞ** (consistent-but-wrong — cache/ledger diverjansı değil, semantik bir aşırı-iddia). `current_state_rebuild_consistency_pre_batch`={signals_checked:270, mismatches_n:0, consistent:True} — tutarlılık kontrolü TEK BAŞINA bu semantik hatayı yakalamaz, ayrı bir kontrol (yukarıdaki §2/§3) gerekiyordu.

### 5. Disposable-only düzeltme
İki parça: (a) `ami/lifecycle/canonical_backfill.py:backfill_lifecycle()` artık HİÇBİR TERMINAL_CLOSE transition yazmıyor (yalnız SIGNAL_BIRTH); (b) yeni `correct_unvalidated_terminal_close()` — gerçek DB'nin (henüz uygulanmamış ama disposable-kopyada tekrarlanan) M-0023 mirasındaki 266 ESKİ TERMINAL_CLOSE satırını append-only bir **CORRECTION** transition'ıyla tersine çeviriyor (`CLOSED→OPEN`, `transition_version=2`, `correction_of=<orijinal transition_id>`, `validate=False` — canonical_schema.py'de zaten belgelenmiş correction-bypass deseni), orijinal satırı SİLMEDEN/DEĞİŞTİRMEDEN. Denormalize `lifecycle_status`/`lifecycle_reason_code` kolonları ledger'la tutarlı kalacak şekilde güncelleniyor (OPEN/CORRECTION).

**Doğrulanan invariant'lar:** 270 signal identity DEĞİŞMEDİ (`identity_unchanged_by_terminal_correction=True`), source_event_id/independent_cycle_id DEĞİŞMEDİ, orijinal 536 satırın TAMAMI hâlâ mevcut ve değişmemiş (`transitions_all_pre_existing_rows_preserved=True` — append-only kanıtı), `event_end_ts_ms` kaybolmadı (ami_events'te sabit), current-state ledger'dan yeniden kuruldu (`rebuild_current_state`), rollback+reapply+idempotency tekrar çalıştı, 478/478 test (472 + 6 yeni) ✓.

### 6. CLOSED/OPEN/TERMINAL/CENSORED sayıları (ÖNCE/SONRA)
Mevcut taksonomide yalnız OPEN/CLOSED implemente — TERMINAL/CENSORED ayrı bir enum değeri olarak YOK (icat edilmedi).
| | ÖNCE | SONRA |
|---|---|---|
| OPEN | 4 | **270** |
| CLOSED | 266 | **0** |
| current_state_rebuild_consistency | consistent=True (ama yanlış) | consistent=True (doğru) |

### 7. Transition tipleri ve sayıları (ÖNCE/SONRA)
| Tip | ÖNCE | SONRA |
|---|---|---|
| SIGNAL_BIRTH→OPEN | 270 | 270 (değişmedi) |
| TERMINAL_CLOSE→CLOSED | 266 | 266 (append-only, silinmedi — artık geçersiz kılınmış tarihsel kayıt) |
| CORRECTION→OPEN | 0 | **266 (yeni)** |
| **Toplam** | **536** | **802** |

### 8. setup-version identity-token verdict
`ami.lifecycle.canonical_schema.UNKNOWN_SETUP_VERSION_TOKEN` eklendi (= `SETUP_VERSION_DEFAULT`="setup-v1", signal_id değişmedi — yalnız isimlendirme/sözleşme netliği). **Supersession contract** önceden belgelendi: gerçek bir setup_version kaynağı bulunursa mevcut signal_id'ler SESSİZCE mutate edilmeyecek — ya (a) yeni `IDENTITY_VERSION` (yeni signal_id, eski korunur) ya da (b) açık supersession kaydı (correction-ledger benzeri) — karar o zaman verilecek, şimdi yalnız "sessiz mutasyon yasak" kısıtı kilitlendi.

### 9. Rollback/idempotency
`rollback_removed_new_objects=True`, `rollback_preserved_lifecycle_tables=True` (post-correction duruma göre karşılaştırıldı — düzeltilen kanon-öncesi duruma göre DEĞİL), `reapply_counts_match_pre_rollback=True`, `row_count_equal_across_reruns=True`, `content_hash_equal_across_reruns=True`. `correct_unvalidated_terminal_close()` 2. çalıştırmada 0 yeni düzeltme (`already_open_or_corrected` ile doğrulandı) — idempotent.

### 10. Full test result
**478/478 ✓** (472 + 6 yeni: `test_fresh_backfill_never_writes_terminal_close_even_with_event_end_ts`, `test_correct_unvalidated_terminal_close_reverses_pre_existing_closed_status`, `test_correct_unvalidated_terminal_close_is_idempotent`, `test_correct_unvalidated_terminal_close_skips_signals_already_open`, `test_unknown_setup_version_token_equals_frozen_default_and_is_hash_stable`, `test_derive_signals_setup_version_column_none_but_identity_hash_uses_token`). `ami/lifecycle/migration_rehearsal.py`'nin ESKİ Phase-7A-canonical-track rehearsal'ı da bu round'un `correct_unvalidated_terminal_close()`'unu çağıracak şekilde güncellendi (reapply-sonrası transition sayısının pre-rollback'ten DOĞAL olarak farklı olması — 270 vs 802 — regresyon değil, "hiç hata yapılmamış taze rebuild" ile "düzeltilmiş miras" arasındaki beklenen asimetri, açıkça raporlandı).

### 11. Gerçek canonical.sqlite doğrulaması (read-only)
`schema_versions.canonical_warehouse=8` (DEĞİŞMEDİ), `ami_signal_lifecycle`=270 satır, `ami_lifecycle_transitions`=536 satır, `lifecycle_status` dağılımı hâlâ CLOSED=266/OPEN=4 (bu batch'in düzeltmesi HENÜZ gerçek DB'ye uygulanmadı — disposable-only), `ami_lifecycle_field_provenance` tablosu gerçek DB'de YOK. Dosya sha256/mtime bu batch içinde değişti ama nedeni bu batch'in kodu DEĞİL — `researcher_exposure_ledger` (986 satır) diğer, önceden var olan Phase 6 research test dosyalarının (W1/W3/W4/W5a/W6/W6rs/W7a/W10a) `feature_gateway`'in zorunlu erişim-denetimi nedeniyle her çalıştırmada gerçek DB'ye yazması — bu batch'ten önce de her tam regresyonda olan, sanksiyonlu, mevcut davranış.

### Protected hashes
SESSION_DELTA_ON_PROTECTED_FILES = ZERO (bu batch'te de hiçbir protected dosyaya Edit/Write yapılmadı).

### Canonical migration readiness verdict
**HAZIR** (v9/16-field/setup_version-NULL/terminal_ts-NULL/TERMINAL_CLOSE-semantic-correction tam disposable doğrulandı) — gerçek DB'ye uygulama (schema v8→v9 + provenance tabloları + setup_version/terminal_ts NULL + 266 CORRECTION transition) yalnız ayrı, açık bir canonical-migration onayıyla yapılabilir. **Not:** gerçek DB'ye uygulanacak düzeltme, gerçek DB'nin ZATEN 266 eski TERMINAL_CLOSE satırı içerdiği (M-0023) gerçeğiyle hesaplaşmalı — bu round'un tasarımı (append-only CORRECTION, asla UPDATE/DELETE) tam olarak bunun için.

```
Disposable semantic-closure round-2 migration executed: YES
Canonical DB migration executed: NO
Canonical lifecycle/transition/provenance rows modified: NO
Observer/binding changed: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

## 68. PHASE 7A-P1 — EFFECTIVE-LEDGER SEMANTICS ROUND 3: TWO-LAYER LEDGER CONTRACT + validate=False FAIL-CLOSED SAFETY (2026-07-04, Sonnet 5)

**Onay:** operatörün son read-only-first talimatı — §67'nin append-only CORRECTION düzeltmesinin (OPEN→CLOSED/TERMINAL_CLOSE→OPEN/CORRECTION) kendisinin gerçek bir close-reopen hareketi OLMADIĞINI, ilk CLOSED transition'ın bilimsel olarak geçersiz olduğunu ve raw-ledger okuyan downstream araştırmanın hâlâ sahte CLOSED interval görebileceğini doğrula, disposable-only düzelt. **CANONICAL DB'YE DOKUNULMADI. NO PHASE 7B. NO RUNTIME/BINDING/OBSERVER CHANGE.**

### 1. CORRECTION transition'ın exact alanları
`previous_status=CLOSED`, `new_status=OPEN`, `reason_code=CORRECTION`, `transition_ts=<orijinal TERMINAL_CLOSE'un transition_ts'i, DEĞİŞMEDİ>`, `known_at_ts=max(transition_ts, düzeltmenin gerçekten uygulandığı wall-clock)`, `correction_of=<orijinal transition_id>`, `transition_version=2`. **recorded_at_ts** — ayrı bir kolon yok; mevcut `created_ms` kolonu (INSERT anındaki wall-clock, `insert_transition()` içinde otomatik) bu rolü zaten karşılıyor, yeni taksonomi icat edilmedi. **idempotency key** — mevcut `(signal_id, previous_status, new_status, transition_ts, reason_code, transition_version, observation_mode)` tuple'ı (→ `transition_id` hash'i) CORRECTION satırları için de AYNEN geçerli, özel bir durum yok.

### 2. Effective lifecycle ledger sözleşmesi — BULGU
Doğrulandı: **öncesinde YALNIZ "en son satırı al" yöntemi vardı** (`rebuild_current_state()`, raw tablo üzerinde `transition_ts,transition_version` sıralamasıyla). `correction_of` ile işaretlenmiş TERMINAL_CLOSE, hiçbir yerde effective timeline/lifecycle-duration sorgularından **dışlanmıyordu** — raw ledger'ı doğrudan okuyan (rebuild_current_state'i hiç çağırmayan) bir downstream sorgu, 266 sinyalin her biri için event_end_ts_ms'e dayalı **sahte bir CLOSED interval** görürdü (bkz. §6 kanıt testi).

### 3. Yeni iki-katmanlı canonical view/sorgu ayrımı
Mevcut repository'de eşdeğer bir "effective ledger" yapısı YOK (reconciliation yapıldı — `ami_lifecycle_transitions` her zaman ham/immutable olarak tasarlanmıştı, `rebuild_current_state()` yalnız STATUS'u ele alıyordu, INTERVAL/DURATION için ayrı bir yüzey yoktu). Paralel ikinci bir truth-layer KURULMADI — yeni `ami_lifecycle_effective_transitions` **VIEW**'i (`ami_lifecycle_transitions` üzerinde, `canonical_schema.py`) eklendi:
- **immutable raw audit ledger** = `ami_lifecycle_transitions` (değişmedi, append-only, artık-geçersiz-kılınmış satırlar dahil HER ŞEY sonsuza dek korunur).
- **effective lifecycle ledger** = `ami_lifecycle_effective_transitions` VIEW — (a) `correction_of` ile başka bir satır tarafından işaretlenmiş (superseded) her satırı VE (b) kendisinin düzelttiği satırın **tam tersi (pure reversal)** olan bir CORRECTION satırını (`correction.previous_status==original.new_status AND correction.new_status==original.previous_status`) dışlıyor — bu ikisi net olarak "hiçbir gerçek transition olmadı" anlamına geliyor. Metadata-only bir correction (AYNI new_status'u yeniden onaylayan) pure-reversal SAYILMIYOR, dışlanmıyor (gerçek bir transition olduğu için korunuyor — test kanıtlı: `test_effective_view_keeps_genuine_non_reversal_correction`).
- Yeni companion fonksiyonlar: `effective_lifecycle_status()` (rebuild_current_state'in effective-view eşdeğeri), `count_effective_closed_signals()` (gerçek/geçerli CLOSED interval'i olan sinyal sayısı).

### 4. Test sonuçları (istenen 8 kontrol)
| Kontrol | Sonuç |
|---|---|
| Effective history'de sahte CLOSED interval | **0** (`count_effective_closed_signals=0`, 266 düzeltilmiş sinyalin hiçbirinde effective CLOSED satırı yok) |
| Signal gerçek anlamda reopened sayılıyor mu | HAYIR — effective view'da yalnız genesis (SIGNAL_BIRTH→OPEN) kalıyor, TERMINAL_CLOSE+CORRECTION çifti tamamen dışlanıyor |
| Lifecycle duration CLOSED→OPEN aralığını hold süresi sayıyor mu | HAYIR — kanıt testi: raw ledger naif sorgusu 266 sinyalin her biri için **sahte 4000ms** (örnek) interval üretir, effective view AYNI sorguda **None/0** döner |
| current-state rebuild | **270 OPEN** (raw: `rebuild_current_state`, effective: `effective_lifecycle_status` — ikisi 270/270 birebir uyumlu, `effective_rebuild_consistency={mismatches_n:0,consistent:True}`) |
| Raw ledger satır sayısı | **802** (immutable, DEĞİŞMEDİ) |
| Effective ledger satır sayısı | **270** (1/sinyal — yalnız genesis) |
| Migration/restart/rerun duplicate correction | **0** (`terminal_close_correction_rerun={signals_corrected:0, already_open_or_corrected:266}`) |
| Correction row gerçek market transition sayılıyor mu | HAYIR — effective view'dan dışlanıyor (pure-reversal kuralı) |

### 5. validate=False safety contract (fail-closed, `insert_transition()`'a eklendi)
- **Yalnız** `reason_code=CORRECTION` **VE** `correction_of` (mevcut bir transition_id'ye referans) birlikte verildiğinde izinli — ikisi de eksikse `LifecycleIntegrityViolation` (fail-closed).
- `correction_of` hedefi ledger'da yoksa → fail-closed (raise).
- Aynı `correction_of` ikinci kez farklı bir correction ile hedeflenirse → raise ("a transition may only be corrected once").
- Aynı correction tuple'ı AYNEN yeniden gönderilirse → mevcut idempotent-noop mekanizması (tid eşleşmesi) hâlâ önce çalışıyor, sessiz no-op.
- Normal transition writer'lar (`backfill_lifecycle`) hiçbir zaman `validate=False` geçirmiyor — yapısal olarak bu bypass'a erişemiyorlar (ek olarak, geçirseler bile `reason_code`/`correction_of` şartları onları engeller). Mevcut bir test (`test_correction_uses_higher_transition_version_not_update`) bu yeni sözleşmeye göre güncellendi (`correction_of=None` → gerçek transition_id).

### 6. Terminal kanıtı olmayan eski sinyaller için canonical query contract
Modül docstring'ine eklendi: **OPEN, "aktif/live" anlamına GELMEZ** — yalnız "terminal durum hiç doğrulanmadı" anlamına gelir. Taksonomiye UNKNOWN/CENSORED/UNRESOLVED icat EDİLMEDİ; bunun yerine MEVCUT `terminal_ts=NOT_IMPLEMENTED` field-provenance kaydı canonical gate olarak kullanılıyor — terminal/hold-duration araştırması önce bu classification'ı VE `ami_lifecycle_effective_transitions`'ı kontrol etmeden hiçbir interval'i geçerli saymamalı (test kanıtlı: `terminal_ts_classifications == {"NOT_IMPLEMENTED"}` tüm 270 sinyal için).

### Raporlanan sayılar (gerçek DB'nin disposable kopyası üzerinde, round 3 sonrası)
raw ledger=802, effective ledger=270, superseded transition=266 (TERMINAL_CLOSE) + 266 (kendini-superseding CORRECTION, pure-reversal olarak dışlanan) = 532 dışlanan, current-state OPEN=270/CLOSED=0, `effective_rebuild_consistency` consistent=True (0 mismatch).

### Full test result
**489/489 ✓** (478 + 11 yeni: `test_validate_false_requires_correction_of`, `test_validate_false_requires_reason_code_correction`, `test_validate_false_fails_closed_on_nonexistent_correction_target`, `test_same_transition_cannot_be_corrected_twice`, `test_identical_correction_resubmission_remains_idempotent_noop`, `test_normal_writer_cannot_reach_validate_false_bypass`, `test_effective_view_excludes_pure_reversal_pair`, `test_effective_view_keeps_genuine_non_reversal_correction`, `test_effective_view_agrees_with_rebuild_current_state_when_no_correction`, `test_effective_view_survives_init_lifecycle_schema_rerun`, `test_naive_raw_ledger_duration_query_is_contaminated_but_effective_is_not`).

### Rollback/idempotency
`ami_lifecycle_effective_transitions` VIEW `rollback_lifecycle_schema()`'ya eklendi (view önce, tablolar sonra düşürülüyor — bağımlılık sırası doğru), reapply'de yeniden oluşuyor. `init_lifecycle_schema()` rerun'da view idempotent (`CREATE VIEW IF NOT EXISTS`). Round-3 içi rerun: `correct_unvalidated_terminal_close()` 2. kez çağrıldığında 0 yeni satır (`raw_ledger_rows_unchanged_after_rerun=True`).

### Gerçek canonical.sqlite doğrulaması (read-only)
`schema_versions=8` DEĞİŞMEDİ, `ami_signal_lifecycle`=270, `ami_lifecycle_transitions`=536 (round 2/3'ün CORRECTION satırları gerçek DB'ye HİÇ yazılmadı), `ami_lifecycle_effective_transitions` VIEW'i gerçek DB'de YOK, `ami_lifecycle_field_provenance` tablosu YOK.

### Protected hashes
SESSION_DELTA_ON_PROTECTED_FILES = ZERO.

### Canonical v9 migration readiness verdict
**HAZIR** — v9/16-field/setup_version-NULL/terminal_ts-NULL/TERMINAL_CLOSE-semantic-correction/effective-ledger-view/validate=False-fail-closed-contract tam disposable doğrulandı, 489/489 yeşil. Gerçek DB'ye uygulama (schema v8→v9 + provenance tabloları + setup_version/terminal_ts NULL + 266 CORRECTION transition + `ami_lifecycle_effective_transitions` VIEW + `insert_transition()`'ın sıkılaştırılmış `validate=False` guard'ları) yalnız ayrı, açık bir canonical-migration onayıyla yapılabilir.

```
Disposable semantic-closure round-3 migration executed: YES
Canonical DB migration executed: NO
Canonical lifecycle/transition/provenance rows modified: NO
Observer/binding changed: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

## 69. PHASE 7A-P CANONICAL PROVENANCE MIGRATION — SCHEMA_DRIFT_BLOCKER, TEST-ISOLATION CLOSURE, RESTORE, CONTROLLED APPLY (2026-07-04, Sonnet 5)

**Onay:** "APPROVE PHASE 7A-P CANONICAL PROVENANCE MIGRATION" → SCHEMA_DRIFT_BLOCKER bulundu, operatör "SELECT OPTION 2 — RESTORE CLEAN V8 AND REAPPLY CONTROLLED MIGRATION" onayıyla devam edildi.

### SCHEMA_DRIFT_BLOCKER (bulundu, sonra kapatıldı)
`ami/warehouse/schema.py`'yi güncelleyip (v9 DDL, disposable-doğrulanmış kodla byte-for-byte aynı) migrasyon script'inden ÖNCE bu batch'in "full regression" gereğini karşılamak için tüm test suite'i çalıştırdım. `test_ami_research_w6rs_confirmation/w6rs_confound_resolution/w7a/w10a/w6_compression_*.py` içindeki "real data smoke test" fonksiyonları `ami.warehouse.schema.connect(DEFAULT_PATH)` (writable) + bazıları `init_schema(conn)` çağırıyor — bu ÖNCEDEN VAR OLAN, meşru bir tasarım (`init_schema` "safe to call on every process start"). Sonuç: gerçek DB'ye migrasyon script'i hiç çalışmadan, yalnız ŞEMA/DDL katmanı (additive, byte-identical, veri DOKUNULMADI) yanlışlıkla uygulandı — schema_version 8→9, 3 yeni obje oluştu, ama 270/536/CLOSED=266,OPEN=4/setup_version NOT NULL hiçbiri değişmedi.

### Aşama 1 — Forensic preservation
- Kaza-eseri v9-schema-only snapshot: `data/ami/backups/canonical_ACCIDENTAL_V9_SCHEMA_DRIFT_snapshot_20260704_161554.sqlite`, sha256=`10583147417867c2e4efc32e946d5e4e6ea382a68c69fa456e1e769aa496f0e8`, schema_version=9, fingerprint=`7f5571fa0ff975d41638a42578e03e01edb5e4f3d1bc9adacce4ac3c1c2e636c`, 270/536/CLOSED=266,OPEN=4, field_provenance=0 satır.
- Temiz v8 backup'a (`canonical_pre_phase7a_p_provenance_migration_20260704_153645.sqlite`) dokunulmadı.
- Migration-code hash'leri kaydedildi (6 dosya, `ami/warehouse/schema.py` + 5 `ami/lifecycle/*.py`).

### Aşama 2 — Test isolation safety closure
**Kök neden kapatıldı, production runtime davranışı DEĞİŞTİRİLMEDİ, yalnız test sınırında çözüldü:**
- `ami/warehouse/schema.py`: `REAL_CANONICAL_PATH_IMMUTABLE` (asla test-time monkeypatch'lenmeyen sabit referans) + `_TEST_ISOLATION_ACTIVE` flag (yalnız conftest.py set eder) eklendi. `connect()`'in `path` parametresi artık `None` default (CALL-TIME'da `DEFAULT_PATH` okunuyor, def-time'da DEĞİL) — bu, conftest'in `DEFAULT_PATH`'i session boyunca yönlendirmesini mümkün kılıyor. Fail-closed guard: `_TEST_ISOLATION_ACTIVE=True` iken gerçek path'e writable bağlantı denemesi `RuntimeError(TEST_ISOLATION_SAFETY_BLOCKER)` fırlatır.
- **Yeni** `tests/conftest.py`: session-scoped + autouse `_isolate_real_canonical_db` fixture — gerçek DB'yi session başında BİR KEZ disposable bir kopyaya (`tmp_path_factory`) kopyalar, `DEFAULT_PATH`'i o kopyaya yönlendirir, `_TEST_ISOLATION_ACTIVE=True` yapar; teardown'da gerçek dosyanın hash/mtime/schema-fingerprint/schema-version'ının session boyunca DEĞİŞMEDİĞİNİ assert eder (değiştiyse TEST_ISOLATION_SAFETY_BLOCKER ile patlar).
- **Yeni** `tests/test_test_isolation_safety.py` (5 test): DEFAULT_PATH yönlendirildi mi, writable-gerçek-path fail-closed reddediliyor mu, read-only hâlâ izinli mi, hash/mtime/fingerprint session içinde sabit mi.
- Doğrulandı: `test_ami_research_w6rs_confirmation/w6rs_confound_resolution/w7a/w10a.py` (real-data + `init_schema` çağıran testler dahil) bu isolation altında çalıştırıldı — gerçek DB'nin mtime'ı **hiç değişmedi**.

### Aşama 3 — Restore
- Aktif writer yok (`BEGIN IMMEDIATE` başarılı) doğrulandı.
- `canonical_pre_phase7a_p_provenance_migration_20260704_153645.sqlite` (sha256=`a3731c9897ccf44c6230b3fea6a423cc7e4f06f3f59d74a6bbf6fd63cab04110`) → gerçek DB'ye restore edildi, restore sonrası hash BİREBİR eşleşti.
- Postflight: schema_version=8 ✓, signal=270 ✓, transition=536 ✓, provenance table/view YOK ✓, fingerprint=`ac12f46083fda7c4ba887245e4d02ce7fb8efd6aaf9848ed0f358bd748f3045e` (orijinal §64 preflight değeriyle birebir) ✓, FORWARD_N=0 ✓, bindings=2 (LONG_HOUR17_HOLD6H, LONG_HOUR17_COMPOSITE) değişmedi ✓. **RESTORE_BLOCKER yok.**

### Aşama 4 — Controlled canonical migration (GERÇEKTEN UYGULANDI)
Disposable'da 495/495 testten geçen EXACT kod, tek script'te, doğru sırayla (`migrate_setup_version_nullable` → `init_schema` [view/tablo bağımlılığı çakışmasını önlemek için, dry-run'da keşfedilen sıralama] → `backfill_lifecycle` → `correct_unvalidated_terminal_close` → `backfill_field_provenance`) çalıştırıldı. **Bulunan ve düzeltilen ek bug (dry-run'da yakalandı, gerçek DB'ye dokunulmadan):** `backfill_lifecycle()`'ın ON CONFLICT UPDATE'i `lifecycle_reason_code`'u her rerun'da `SIGNAL_BIRTH`'e sıfırlıyordu — `correct_unvalidated_terminal_close()`'un işaretlediği `CORRECTION`'ı siliyordu. Düzeltme: bu iki kolon ON CONFLICT UPDATE'ten çıkarıldı (ledger/correction mekanizmasının sahipliğinde kalacak şekilde); yeni regression testi eklendi.

### Gerçek DB sonucu (uygulandı, doğrulandı)
- schema_version: **9**, fingerprint: `ef8eeebb10069f21450c3cfcd556e9723540817795d58fd34237907badd6dcb8`
- lifecycle signals: **270**, raw transitions: **802** (270 SIGNAL_BIRTH + 266 TERMINAL_CLOSE + 266 CORRECTION), effective transitions: **270**, effective CLOSED interval: **0**
- current state: **270 OPEN** (raw+effective birebir uyumlu, 0 mismatch)
- provenance: **4320** satır, 0 eksik, 0 duplicate; direction 270/270 HISTORICAL_PROXY; setup_version/terminal_ts 270/270 NULL+NOT_IMPLEMENTED; first_executable_ts NULL+FORWARD_ONLY
- FORWARD_N: 0→0 (değişmedi), bindings: 2→2 (değişmedi)
- 2. çalıştırma (idempotency): 0 yeni satır, content hash birebir aynı (`54e13f6d9885d0cf0199098b6bb92840c7244f1eee4aaf66c3ed87c98fac74b0`)
- old-reader (`feature_gateway.fetch_events`): 252 — etkilenmedi
- post-migration backup: `data/ami/backups/canonical_post_phase7a_p_provenance_migration_v9_20260704_162055.sqlite`, sha256=`3d34f44ef799ff4fb61badf2220fe4c6a14b33ca547ea0399a03b796e0541bc2`

### Aşama 5 — Postflight (isolated test DB'lerde tam regresyon)
495/495 test (37 dosya + buyfade/silexit + yeni `test_test_isolation_safety.py` [5] + yeni `test_backfill_rerun_after_correction_does_not_clobber_reason_code` [1]) çift-çift sıralı çalıştırıldı — **hepsi ✓**. 2 test (`test_ami_lifecycle_provenance_rehearsal.py::test_full_provenance_rehearsal_real_data`, benzer mantık) migrasyon artık kalıcı olduğu için (`schema_version_before` artık 8 değil 9) her iki duruma dayanıklı hâle getirildi (`test_schema_fingerprint_changes_only_by_addition` emsaliyle). **Gerçek canonical.sqlite'ın hash/mtime'ı bu TÜM postflight regresyon boyunca BİREBİR SABİT kaldı** (`3d34f44e...`/`1783171183.22423`, çalıştırma öncesi/sonrası doğrulandı) — TEST_ISOLATION_SAFETY_BLOCKER hiç tetiklenmedi.

### Protected hashes
SESSION_DELTA_ON_PROTECTED_FILES = ZERO.

### Unresolved blockers
YOK.

### Phase 7A closure verdict
**KAPANDI.** Şema v9, 16-field provenance, setup_version/terminal_ts NULL semantiği, 266 append-only CORRECTION, effective-ledger view, validate=False fail-closed contract — hepsi gerçek `data/ami/canonical.sqlite`'a uygulandı ve doğrulandı. Test-isolation safety closure kalıcı bir altyapı parçası olarak kaldı (`tests/conftest.py` + `ami/warehouse/schema.py`'nin `REAL_CANONICAL_PATH_IMMUTABLE`/`_TEST_ISOLATION_ACTIVE`/`connect()` guard'ı) — gelecekteki HERHANGİ bir test çalıştırması artık yapısal olarak gerçek DB'ye yazamaz.

### Phase 7B readiness verdict
**Phase 7B'ye geçilmedi, otomatik geçiş yok.** Phase 7A'nın (identity+lifecycle+provenance) tamamlanmış olması Phase 7B'nin (timing/path/lifecycle-transition ENGINE — NO_TIMING_PATH_ENGINE bu batch'te de yasaktı) ayrı, açık bir onay gerektirdiği gerçeğini değiştirmez.

```
Canonical v9 migration executed: YES
Canonical provenance backfill executed: YES
Canonical corrections appended: YES
Observer started: NO
Existing binding changed: NO
FORWARD_N increased: NO
Order/live/shadow behavior changed: NO
```

WAIT_FOR_OPERATOR_APPROVAL

## 70. PHASE 7B — PATH/MFE/MAE OBSERVATION ENGINE: DISPOSABLE FOUNDATION + SEMANTIC CLOSURE + CANONICAL MIGRATION (2026-07-04, Sonnet 5)

Üç operatör onayı, tek oturumda: "APPROVE PHASE 7B-0 DISPOSABLE PATH ENGINE FOUNDATION — WITH SEMANTIC LOCKS" → "APPROVE PHASE 7B-0.1 DISPOSABLE SEMANTIC CLOSURE" → "APPROVE PHASE 7B CANONICAL PATH METRICS MIGRATION". §62'nin read-only 7A→7B→8A→8B tasarım denetimini (Phase 7A KAPANDI, §69) takip eden ilk kod-yazan Phase 7B batch'i.

### 7B-0 — Disposable foundation
Yeni `ami/lifecycle/{path_schema,path_metrics,path_migration_rehearsal}.py`, DISPOSABLE_DB_ONLY. Operatörün 10 semantic-lock'u uygulandı: candle-boundary (birth'i saran kısmi candle hem reference hem path'ten hariç), MFE≥0/MAE≤0, 1m-proxy timing, `horizon_outcome_class` adı (W4'ün `classify_path()` birebir reuse — paralel taksonomi yok), ayrı vol-units alanları, 30-kolon şema, 6-durumlu status taxonomy, signal-level satır + independent_cycle_id denominatör disiplini, W4'ün 4 sabit ufku (scalp_30m/1h, swing_4h/24h) DIŞINDA yeni ufuk yok. **Disposable sonuç:** 1080 satır (270×4), OK=912/MISSING_INTERNAL_GAP=153/INVALID_VOLATILITY_BASELINE=2/EXCLUDED_NO_HORIZON_DATA=13. 49 yeni test — toplam AMI 544/544.

### 7B-0.1 — Semantic closure
6 düzeltme uygulandı (detay: BATCH-P7B-0.1 ledger satırı):
1. `observation_status` (path-only, 5 değer) / `volatility_status` (OK/INVALID_VOLATILITY_BASELINE/NOT_APPLICABLE) AYRILDI, DB CHECK ile eşleştirildi.
2. Sıfır-ekstremum timing donduruldu: t=0 referans noktası path'in parçası, gerçek path hiç favorable/adverse'a ulaşmazsa `time_to_mfe/mae_ms=0` (ilk candle ts'i sessizce kullanılmıyor).
3. Yeni `ami/lifecycle/path_field_provenance.py` — 23 `path_observations.*` alan, MEVCUT `ami_lifecycle_field_provenance` tablosuna yazıyor (paralel tablo yok), proxy→safe downgrade guard'ı kodlu.
4. Tam şema manifest üretildi (31 kolon — 30 operatör-spec + `volatility_status`; önceki 7B-0'ın "30 kolon" iddiası `PRAGMA table_info` ile doğrudan sayılıp teyit edildi, hatalı bir "29" iddiası hiç yapılmamıştı).
5. `*_vol_units`→`*_anchor_vol_units` (Seçenek A): payda (`realized_vol_at_anchor`) ufuktan bağımsız sabit bir anchor-window ölçüsü — isim bunu netleştiriyor, ayrı bir `normalization_basis` kolonu yerine.
6. Tüm disposable doğrulama tekrar çalıştırıldı. **Disposable sonuç:** observation_status OK=914/MISSING_INTERNAL_GAP=153/EXCLUDED_NO_HORIZON_DATA=13; volatility_status OK=912/NOT_APPLICABLE=166/INVALID_VOLATILITY_BASELINE=2 (7B-CANON'un gerçek-veri sonucuyla BİREBİR aynı — disposable doğrulama gerçek veriyi birebir yansıtıyordu). 80 yeni test (49'un yerine) — toplam AMI 575/575.

### 7B-CANON — Canonical migration (GERÇEK `data/ami/canonical.sqlite`'A UYGULANDI)

**Preflight (v9 checkpoint doğrulaması):**
```
schema_version=9, signals=270, raw_transitions=802, effective_transitions=270,
field_provenance=4320, FORWARD_N=0, bindings=2 (LONG_HOUR17_HOLD6H/LONG_HOUR17_COMPOSITE)
active_writer=NONE (BEGIN IMMEDIATE succeeded)
```
Hepsi operatörün beklediği checkpoint'le BİREBİR eşleşti — PHASE_7B_MIGRATION_PREFLIGHT_BLOCKER TETİKLENMEDİ.

**Backup:** `data/ami/backups/canonical_pre_phase7b_path_metrics_migration_20260704_180134.sqlite`, sha256=`3d34f44ef799ff4fb61badf2220fe4c6a14b33ca547ea0399a03b796e0541bc2` (kaynakla birebir doğrulandı), fingerprint=`ef8eeebb10069f21450c3cfcd556e9723540817795d58fd34237907badd6dcb8`.

**Canonical değişiklikler:**
1. `ami/warehouse/schema.py`: `CANONICAL_SCHEMA_VERSION` 9→10, yeni `_SCHEMA_PHASE7B` (path_schema.py'nin `_SCHEMA`'sının fully-interpolated hâlinin byte-for-byte kopyası — fresh in-memory DB'de fingerprint `3a26ffa86ecec9d8b63eff9455e3cfbbd594cc59eb36896feeba4d3bf232f1e7` ile doğrulandı, disposable-validated değerle BİREBİR).
2. Yeni `ami/lifecycle/path_canonical_migration.py:run_canonical_migration()` — `freeze_and_record()` + `backfill_path_field_provenance()`'ı tek auditable entry point'te birleştiriyor.
3. `ami/research/feature_gateway.py`: `KNOWN_FEATURE_TABLES`'a 3 tablo eklendi (`ami_signal_lifecycle`/`ami_lifecycle_effective_transitions`/`ami_lifecycle_path_observations`); 3 yeni dedike fetch fonksiyonu — `fetch_lifecycle_signals()` (curated kolon + REAL/PROXY pooling guard, `EvidenceLayer`'ın kendi vocabulary'si, `assert_not_pooled`'ın `SourceQuality`'siyle KARIŞTIRILMADI), `fetch_lifecycle_effective_transitions()` (raw `ami_lifecycle_transitions` DEĞİL effective view — bu, raw TERMINAL_CLOSE'un asla terminal-evidence olarak sunulmamasının gateway-seviyesi garantisi), `fetch_path_observations()` (equals allowlist). `fetch_chart_feature()` bu 3 tabloyu reddediyor (dedike fonksiyon zorunlu).
4. Backfill: 270 sinyal × 4 ufuk = **1080 path satırı**; 270×23=**6210 yeni field-provenance satırı**.

**Backfill sonucu (operatörün beklediğiyle BİREBİR eşleşti):**
```
observation_status: OK=914, MISSING_INTERNAL_GAP=153, EXCLUDED_NO_HORIZON_DATA=13,
                     MISSING_REFERENCE_PRICE=0, NOT_COMPUTABLE_DIRECTION=0
volatility_status:  OK=912, INVALID_VOLATILITY_BASELINE=2, NOT_APPLICABLE=166
provenance: 6210 yeni + 4320 mevcut = 10530 toplam, 0 eksik, 0 duplicate, 0 proxy→safe downgrade
```

**Postflight:** schema_version=10, signals=270 (değişmedi), raw_transitions=802 (değişmedi), effective_transitions=270 (değişmedi), effective_CLOSED=0, path_rows=1080, total_provenance=10530, FORWARD_N=0 (değişmedi), bindings=2 (aynı signal/knowledge_id çiftleri, değişmedi). 2. çalıştırma (idempotent rerun, doğrudan gerçek DB'ye karşı): row count + content hash + provenance count birebir aynı.

**Mandatory semantic assertions (SQL ile gerçek veriye karşı doğrulandı, 0 ihlal):**
```
reference_price_ts <= signal_birth_ts:                      0 ihlal
effective_path_start_ts >= signal_birth_ts:                  0 ihlal
known_at_ts >= horizon_end_ts:                                0 ihlal
as_of_ts >= known_at_ts (EXCLUDED_NO_HORIZON_DATA hariç):     0 ihlal
mfe_bps < 0:                                                  0 ihlal
mae_bps > 0:                                                  0 ihlal
mfe_bps=0 ama time_to_mfe_ms≠0:                                0 ihlal
mae_bps=0 ama time_to_mae_ms≠0:                                0 ihlal
time_to_mfe==time_to_mae ama SAME_CANDLE_UNKNOWN değil:        0 ihlal
mfe_bps/mae_bps field_classification:                         HISTORICAL_PROXY (270/270)
realized_vol_at_anchor field_classification:                  DETERMINISTIC_HISTORICAL_SAFE
count_effective_closed_signals():                              0
lifecycle_status dağılımı:                                    {OPEN: 270} (270 OPEN ≠ aktif pozisyon)
```
NO_EXIT/NO_STOP/NO_CANCELLATION/NO_REENTRY/NO_TERMINAL_TIME hiçbir yerde türetilmedi — path_canonical_migration.py yalnız FROZEN sabit-ufuk ölçümü yazdı.

**Yan bulgu+düzeltme (gerçek DB'ye dokunmadan):** `ami/lifecycle/provenance_rehearsal.py`'nin `provenance_row_counts()`/`provenance_content_hash()`'i `ami_lifecycle_field_provenance` üzerinde scope'suz `COUNT(*)`/`SELECT *` yapıyordu (Phase 7A-P1'de yazılmıştı, o zaman tablonun TEK yazıcısı kendisiydi) — path_field_provenance artık aynı tabloyu meşru şekilde paylaşınca (`path_observations.*` namespace) 4320 varsayımı gerçek 10530 ile çakıştı. `provenance_version=?` filtresi eklenerek düzeltildi (gerçek bug, kötü niyetli değişiklik değil — paylaşımın ortaya çıkardığı). `tests/test_ami_lifecycle_provenance_rehearsal.py`'nin `schema_version_before in (8,9)` kontrolü `(8,9,10)`'a genişletildi (v9→v10 lifecycle semantiğini DEĞİŞTİRMEDİ, emsal: `test_schema_fingerprint_changes_only_by_addition`).

**Full regresyon (isolated test DB'lerde, 46 dosyanın TAMAMI tek tek/çift-çift çalıştırıldı):**
```
Existing 495 (Phase 7A/7A-P) + 13 yeni gateway testi + 80 yeni path-modülü testi = 588/588 ✓
Gerçek canonical.sqlite hash/mtime bu regresyon boyunca BİREBİR SABİT kaldı:
  test-öncesi:  sha256=fa57cc79f84fbf49d6d84d5897f1dd63230de3a2d935c2329f5f44d8200b47f1, mtime=1783177712.899
  test-sonrası: sha256=fa57cc79f84fbf49d6d84d5897f1dd63230de3a2d935c2329f5f44d8200b47f1, mtime=1783177712.899
```
Post-migration backup: `data/ami/backups/canonical_post_phase7b_path_metrics_migration_v10_20260704_180906.sqlite`, sha256=`fa57cc79f84fbf49d6d84d5897f1dd63230de3a2d935c2329f5f44d8200b47f1`.

**Protected session delta:** ZERO — `tools/s34_state_machine_live_executor.py` sha256=`a8067f2c...` (değişmedi), `tools/s34_realtime_shadow_runner.py` sha256=`a326a5c4...` (değişmedi), `execution/`/`risk/`/`brain/`/`.env` dokunulmadı.

**Unresolved blockers:** YOK.

### Phase 7B canonical-foundation closure verdict
**KAPANDI.** `ami_lifecycle_path_observations` (schema v10) + field-level provenance (10530 satır) gerçek `data/ami/canonical.sqlite`'a uygulandı ve tam doğrulandı. `ami/research/feature_gateway.py` lifecycle/path erişimi için 3 yeni dedike fonksiyonla genişletildi (exposure-logging + known-at enforcement korundu).

### Timing/path research readiness verdict
Path/MFE/MAE ölçüm katmanı artık CANONICAL ve sorgulanabilir (feature_gateway üzerinden) — ama bu, W8/W9/W10-transition/W12'nin OTOMATİK yeniden açılması anlamına GELMEZ (Protocol §5, §69'un kendi closure maddesi aynen geçerli):
- **W9 BAD_TIMING** — hâlâ kendi OD-008 onayını gerektirir; 7B bu path/vol-normalizasyon sözlüğünü sağladı ama yeniden-açma yetkisi VERMEDİ.
- **W8/W12** — hâlâ NOT_IMPLEMENTED (ayrı, inşa edilmemiş proxy-cascade taksonomisi gerektiriyor; 7B bu boşluk için hiçbir şey yapmadı).
- **W10-transition (LONG↔SHORT)** — hâlâ BLOCKED_BY_DATA (OD-017): 7B'nin popülasyonu artık hem LONG(220) hem SHORT(50) path-ölçümlü sinyal içeriyor, ama "transition" whitepaper anlamında TEK bir anchor'ın yapısal yönünün zamanla değişmesi demek — 7B bu veriyi YARATMADI, OD-017 açık kalıyor.
- **Yeni açılan (ama OTOMATİK DEĞİL, kendi preregistration'ını gerektiren) olası soru:** "aynı cascade popülasyonunda LONG_SILENCE(220) ile SHORT_NEITHER/BUY_FADE_SHORT(50) arasında MFE/MAE farkı var mı" — bu YENİ bir hipotez (Protocol §2), mezarlıktan yeniden açılan değil; 7B yalnız ölçüm altyapısını sağladı, testi ÇALIŞTIRMADI.
- Sonraki aday (operatör onayına bağlı, bu batch'te BAŞLATILMADI): 8A (forward_pipeline genişletme + önce eksik test kapatma) veya path/MFE-MAE verisi üzerinde ilk descriptive research dalgası — ikisi de ayrı, açık bir onay gerektirir (NO_AUTOMATIC_TIMING_RESEARCH_WAVE_START).

## 71. W8-HOLD-BASELINE — FIXED-HORIZON MFE/MAE BASELINE + CHRONOLOGICAL STABILITY (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE W8-HOLD-BASELINE-001 — WITH FINAL SEMANTIC LOCKS". §70'in Phase 7B-1 reconciliation'ının önerdiği "en küçük savunulabilir ilk batch" — W8'in kendi zorunlu ön-koşulu (hold-baseline benchmark), competing-risk/management hipotezi DEĞİL.

**NOT_A_MANAGEMENT_WAVE:** hiçbir stop/exit/partial-exit/time-stop/re-entry/cancellation kuralı test edilmedi veya simüle edilmedi; `ami/research/w8_hold_baseline.py` yalnız `ami_lifecycle_path_observations`'ta ZATEN hesaplanmış mfe_bps/mae_bps'i karakterize etti.

### Popülasyon (frozen, feature_gateway üzerinden)
`ami_lifecycle_path_observations` WHERE `observation_status='OK'`, `ami_signal_lifecycle.direction`/`signal_birth_ts`/`source_event_id`/`independent_cycle_id` ile Python'da join edildi (`fetch_lifecycle_signals`+`fetch_path_observations`, raw SQL join YOK).

```
raw_signal_n_population=266 (4 sinyal hiçbir ufukta OK almadı — dürüstçe raporlandı)
distinct_source_event_n_population=248
distinct_independent_cycle_n_population=167 (primary denominator)
```

### 16-hücre family (2 metrik [mfe_bps/mae_bps] × 4 ufuk × 2 yön), chronological 70/30 (signal_birth_ts)

| | LONG (8 hücre) | SHORT (8 hücre) |
|---|---|---|
| N aralığı | 123-216 | 32-50 |
| cycle_n aralığı | 86-142 | 23-40 |
| **Verdict** | **8/8 `ANSWERED_SUPPORTED_STABLE_BASELINE`** | **8/8 `INSUFFICIENT_SAMPLE`** |

SHORT'un TÜM 8 hücresi (yalnız swing_24h değil) MIN_BUCKET_N=20 kuralını 70/30 split sonrası karşılamadı (TEST split ≤15) — operatörün "SHORT swing_24h beklenen insufficient" öngörüsünden DAHA GENİŞ çıktı, ama aynı frozen kuralın dürüst, birleştirilmemiş sonucu (SHORT verisi hiçbir hücrede LONG ile veya başka bir ufukla harmanlanmadı).

`classify_cell_verdict()` üç etiketten HİÇBİRİNİ otomatik atamadı — CI/Holm anlaşmazlığı durumunda (operatörün tanımlamadığı 4. dal) her zaman muhafazakâr `ANSWERED_REGIME_DEPENDENT_BASELINE`'a düştü, asla sessizce "stable"a yuvarlanmadı (test: `test_classify_cell_verdict_never_returns_answered_supported_bare`).

### Eşleştirilmiş negatif kontrol (`ami_candidate_universe`, is_event_aligned=0)
Ay/session/vol-bucket stratified sampling, OUTCOME OKUNMADAN ÖNCE dondu: **266/266 hedef, 0 shortfall** (her stratum tam karşılandı). LONG/SHORT direction-ratio eşleştirilemedi (rastgele bir zaman dilimi gerçek bir yön kararı taşımıyor) — `direction_ratio_matching_status`/`primary_16cell_comparison_status` = **`BLOCKED_FOR_DIRECTION_MATCHING`**, LONG'a varsayılmadı, havuzlanmadı. Ayrı, açıkça `upside_excursion_bps`/`downside_excursion_bps` adlı (asla `mfe_bps`/`mae_bps` değil) bağlamsal istatistik raporlandı:
```
scalp_30m: n=261, upside_med=28.62bps, downside_med=-26.98bps
scalp_1h:  n=257, upside_med=41.11bps, downside_med=-41.16bps
swing_4h:  n=226, upside_med=89.41bps, downside_med=-72.36bps
swing_24h: n=147, upside_med=236.42bps, downside_med=-177.96bps
```

### Mezarlık farkındalığı doğrulaması
`failure_archive` #3 (tight vol-scaled stops), #4 (partial exits), #10 (loser time-stops), #13 (MFE50 giveback single-feature), #18 (BUYFADE management overlay) — hiçbiri bu modülde test EDİLMEDİ (statik kod-taraması testiyle kilitlendi: `test_no_graveyarded_management_rule_terms_in_module_source`, forbidden-terms listesi: stop_loss/partial_exit/time_stop/re_entry/cancellation_rule/management_rule/take_profit/trailing_stop — 0 hit). `BUY_FADE_SHORT_H45_SL75` (19/270 sinyal, #16/#18/#19/#21 mezarlık ailesinin popülasyonunda) hiçbir özel işlem görmedi — diğer sinyallerle AYNI genel formülle okundu.

### Performans bug'ı (bulunup düzeltildi, davranış aynı)
`ami/lifecycle/path_metrics.py:_CandleOHLCIndex.path_window()`'ın `self._rows[lo:]` sınırsız slice'ı — Python bu slice'ı O(n) kopyalıyordu (173,464 satırlık candle index), 270 kanonik sinyal için (çoğu tarihin GEÇ kısmında, `lo` büyük, slice küçük) tesadüfen ucuz kalmıştı, ama W8'in eşleştirilmiş kontrolü TÜM tarihe yayılı ~2900 candidate-universe slotunu (bazıları `lo≈0`, slice≈173K) probe edince ~10+ dakikaya kadar yavaşladı. `bisect.bisect_right` ile sınırlı `self._rows[lo:hi]`'a düzeltildi — matematiksel olarak ÖZDEŞ sonuç (hem `_open_ts` hem `_close_ts` aynı satır-sırasında monoton artan, kanıt: 60/60 path_metrics/path_field_provenance/path_migration_rehearsal testi DEĞİŞMEDEN geçti), yalnız performans iyileşti (real-data rehearsal ~40s→~22s, W8'in kendi gerçek-DB koşumu ~5.7s).

### Gerçek DB çalıştırması
Preflight: aktif yazıcı yok (`BEGIN IMMEDIATE` başarılı), pre-counts (270/802/1080/10530) doğrulandı. `run canonical.sqlite`'a **yalnız** `experiment_registry`(1 satır, E-W8-HOLD-BASELINE-001)+`experiment_results`(20 satır: 3 popülasyon-metadata+negative_control+16 hücre) yazıldı. Postflight: `ami_signal_lifecycle`/`ami_lifecycle_transitions`/`ami_lifecycle_path_observations`/`ami_lifecycle_field_provenance` sayıları (270/802/1080/10530) ve `schema_version`(10) DEĞİŞMEDİ. 2. çalıştırma (idempotent rerun, doğrudan gerçek DB'ye karşı): `experiment_results` satır sayısı (20) + tüm hücre medyanları birebir aynı.

### Full regresyon
608/608 ✓ (588 önceki + 20 yeni W8 testi) — tüm 47 dosya tek tek/çift-çift, isolated test DB'lerinde. Gerçek canonical.sqlite hash/mtime bu regresyon boyunca BİREBİR SABİT:
```
test-öncesi:  sha256=2f356bb843a709d3ad1831b2d0e2eee21ed31e5e7883e19da035091906fe760a, mtime=1783180279.645
test-sonrası: sha256=2f356bb843a709d3ad1831b2d0e2eee21ed31e5e7883e19da035091906fe760a, mtime=1783180279.645
```

### Protected session delta
ZERO — `tools/s34_state_machine_live_executor.py` sha256=`a8067f2c...` (değişmedi), `tools/s34_realtime_shadow_runner.py` sha256=`a326a5c4...` (değişmedi).

### Unresolved blockers
YOK.

### Next-wave eligibility verdict
- **W8 kendisi (competing-risk/management hipotezi):** hâlâ AÇILMADI. Hold-baseline artık mevcut (bu batch), ama gerçek bir management-rule testi #3/#4/#10/#13/#18 mezarlığıyla aynı zeminde olur — yeni retry-condition veya genuinely yeni bir feature/mekanizma olmadan açılmamalı.
- **W9** — hâlâ BLOCKED (OD-008, değişmedi).
- **W10-transition** — hâlâ BLOCKED_BY_DATA (OD-017, değişmedi).
- **W12** — hâlâ transitively BLOCKED (W8'in management-rule kısmına bağımlı).
- **SHORT-yön genel gözlemi:** bu popülasyonda SHORT sinyal sayısı (32-50/hücre) 70/30 split + MIN_BUCKET_N=20 disipliniyle STATISTIK OLARAK yetersiz kalıyor — bu, gelecekteki HERHANGİ bir SHORT-yön chronological-split analizi için bir yapısal sınırlama olarak not edilmeli (fabrikasyonla "çözülmez", yalnız daha fazla veri veya farklı bir tasarımla).

**Kodlamaya başlanmadı** (bu ifadeden sonra) — batch TAMAMLANDI, bir sonraki adım için operatör onayı bekleniyor.

WAIT_FOR_OPERATOR_APPROVAL

## 72. NEXT RESEARCH FRONTIER RECONCILIATION + W8-VOL-NORMALIZED-BASELINE (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE NEXT RESEARCH FRONTIER RECONCILIATION" → read-only bulgular sunuldu → "APPROVE W8-VOL-NORMALIZED-BASELINE-001" → Candidate E çalıştırıldı.

### Read-only reconciliation bulguları (canonicalize EDİLMEDİ, yalnız raporlandı)
- **Shadow ledger büyümesi:** `ami.identity.shadow_ledger_ingest.parse_shadow_ledger()` gerçek (canlı çalışan) ledger'a karşı fresh çalıştırıldığında **262 anchor** üretiyor, canonical `ami_events`'te ise **252** var — 10 anchor'lık fark TAMAMEN SHORT-yön route'larda (BUY_FADE_SHORT_H45_SL75 +9, SHORT_NOISY_BTC1M_D5_H180 +1). "Daha fazla tarihsel veri zaten birikti" doğrulandı, ama YETERSİZ (SHORT 50→~60 büyürdü, 70/30 split için gereken N≥67'yi hâlâ karşılamıyor).
- **SHORT_NOISY/SHORT_CONVICTION ailesi:** `tools/research_s34_short_conviction.py`/`research_s34_short_score3_gauntlet.py` (BTC-liq-anchored SHORT, ETH SELL≥200K→BTC SELL confirm) — N=7-71 (eşiğe göre), 4.4 ay, bazı hücrelerde WR%70-100/mc_p<0.05. `knowledge.sqlite` kontrol edildi: **0 Knowledge Object** bu aileye referans veriyor — hiç mezarlıklanmamış, hiç yönetişime girmemiş, bağımsız/loose bir araştırma artifact'ı. SHORT örneklem sorununu YAPISAL olarak çözebilecek en güçlü aday, ama Phase 3 identity/dedup/R-09/cycle-resolver disiplininden hiç geçmemiş — tam bir canonicalization dalgası gerektirir.
- **`StructurePhase.RECLAIM`:** `ami/enums.py`'de zaten var olan bir state-primitive (`ami/states/engine.py._structure()` zaten üretebiliyor), ama hiçbir event_family "RECLAIM'e geçiş" anını anchor olarak kullanmıyor. W10-transition'ı (OD-017) çözebilecek en doğrudan yapısal yol — `failure_archive` #20 (BUYFADE S→L flip, INSUFFICIENT_SAMPLE, mezarlıklanmamış) ile ilişkili ama dar kapsamlı.
- W8(management)/W9/W10-transition/W12 durumları DEĞİŞMEDİ (zorla açılmadı).

### W8-VOL-NORMALIZED-BASELINE (Candidate E) — çalıştırıldı
Aynı 16-hücre tasarımın (W8-HOLD-BASELINE, E-W8-HOLD-BASELINE-001) BİREBİR reuse'u — `ami/research/w8_hold_baseline.py`'nin `_cell_rows`/`compute_cell`/`classify_cell_verdict`/cluster-bootstrap/permütasyon/Holm fonksiyonları import edildi, yeniden yazılmadı. Yalnız: METRICS=(`mfe_anchor_vol_units`,`mae_anchor_vol_units`), popülasyon ek olarak `volatility_status='OK'` gerektiriyor.

**Popülasyon:** raw_signal_n_population=265 (W8-HOLD-BASELINE'ın 266'sından 1 az — 1 sinyalin tek OK satırı volatility-invalid), distinct_source_event_n=247, distinct_independent_cycle_n=167 (aynı tavan).

**16 hücre — LONG (8/8):** hepsi `ANSWERED_SUPPORTED_STABLE_BASELINE` (n=123-215, cycle_n=86-142) — **0 hücre regime-dependent'e DÜŞMEDİ**. **SHORT (8/8):** hepsi yine `ANSWERED_SUPPORTED_STABLE_BASELINE` DEĞİL, `INSUFFICIENT_SAMPLE` (n=32-50) — normalizasyon örneklem boyutunu ONARMADI (beklenen, dürüstçe teyit edildi, null/failure olarak yeniden-yorumlanmadı).

**Raw-bps karşılaştırması** (`E-W8-HOLD-BASELINE-001`'in stored sonuçlarına karşı, ast.literal_eval ile read-only okundu, yeniden hesaplanmadı):
```
any_long_cell_stable_to_regime_dependent: []   (hiçbiri değişmedi)
normalization_repairs_sample_size: False        (SHORT 8/8 hâlâ INSUFFICIENT_SAMPLE)
```
train-test/IQR oran karşılaştırması (birim-bağımsız effect-size proxy, çünkü vol-normalizasyon satır-başına bölme olduğundan bps farkının basit bir yeniden-ölçeklemesi DEĞİL): LONG hücrelerde karışık yönde ama küçük (ör. scalp_30m LONG: 0.101→0.0119 azaldı; swing_4h LONG: 0.0135→0.0377 hafif arttı; swing_24h LONG: 0.2705→0.2848 hafif arttı) — hiçbiri "stable" sınıflamasını bozacak büyüklükte değil.

**Full regresyon:** 621/621 ✓ (608 önceki + 13 yeni). Gerçek canonical.sqlite hash/mtime bu regresyon boyunca BİREBİR SABİT (`bdb8cf88...` / `1783182347.086`, test-öncesi=test-sonrası). Canonical tablolar (270/802/1080/10530, schema_version=10) DEĞİŞMEDİ — yalnız `experiment_registry`(+1)/`experiment_results`(+20) yazıldı. Protected session delta ZERO.

### Unresolved blockers
YOK.

### Next-frontier readiness verdict
- W8(management)/W9/W10-transition/W12 hâlâ AÇILMADI (değişmedi).
- SHORT örneklem sınırlaması hâlâ YAPISAL — ne normalizasyon ne de mevcut forward-birikim (10 anchor) tek başına çözüyor.
- İki somut, gelecekteki (bu batch'te BAŞLATILMAYAN) canonicalization adayı belgelendi: SHORT_NOISY/SHORT_CONVICTION ailesi (SHORT N'yi büyütebilir) ve RECLAIM-anchored transition ailesi (OD-017'yi çözebilir) — ikisi de kendi ayrı, operatör onaylı identity-canonicalization dalgasını gerektirir.

**Kodlamaya başlanmadı** (bu ifadeden sonra) — batch TAMAMLANDI.

WAIT_FOR_OPERATOR_APPROVAL

## 73. SHORT_NOISY IDENTITY-CANONICALIZATION RECONCILIATION + KNOWN-AT OVERLAP + DISPOSABLE REHEARSAL (2026-07-04, Sonnet 5)

Operatör onay zinciri: "APPROVE SHORT_NOISY IDENTITY-CANONICALIZATION RECONCILIATION — READ-ONLY DESIGN ONLY" → "APPROVE SHORT_NOISY KNOWN-AT OVERLAP RECONCILIATION — READ-ONLY / NO CANONICALIZATION" → "APPROVE SHORT_NOISY KNOWN-AT V1 DISPOSABLE CANONICALIZATION REHEARSAL". İlk iki adım READ_ONLY_ONLY/NO_CANONICAL_WRITE (§72'nin bulduğu SHORT_NOISY/SHORT_CONVICTION adayının forensic derinlemesine incelemesi) — bu iki read-only adım kod/DB üretmedi, yalnız bulgu; üçüncü adım (bu bölümün asıl konusu) disposable rehearsal kodu üretti.

### Read-only forensic bulgular (özet, canonicalize EDİLMEDİ)
- `tools/research_s34_short_conviction.py`'nin `run_S5` karşılaştırması kanıtladı: eski `SHORT_NOISY_BTC1M_D5_H180` ailesi entry noktası olarak **noisy_ts**'i kullanıyordu, ama filtre koşulu (BTC confirmation) noisy_ts'ten SONRA gerçekleşiyordu — lookahead (`S5_entry_noisy` +91.0bps mc_p=0.012 vs `S5_entry_confirm` +32.6bps mc_p=0.176, anlamsız). Eski WR/p-value hiçbir zaman kanıt olarak yeniden kullanılmadı.
- `reconstruct_anchors()` (613 ham anchor) ile canonical `ami_events` (252) arasındaki örtüşme: 234/252 (%93) canonical anchor'ın reconstructed karşılığı var; 234/613 (%38) reconstructed anchor canonical'e denk geliyor — **iki farklı payda, karıştırılmadı** (operatör düzeltmesi).
- Known-at-safe frozen tanım donduruldu: kimlik kaynağı=mevcut canonical `ami_events` (yeni event/family YOK), noisy follow-on=ilk ETH SELL≥50K (anchor+1m,+30m], BTC confirmation=ilk BTC SELL≥200K (noisy_ts+5m,+30m], signal_birth_ts=conf_ts (ASLA noisy_ts).
- Overlap funnel: 252 anchor→164 noisy→**54 BTC-confirmed** (nihai aday popülasyon). 40 distinct cycle dokunuldu; **cycle-seviyesinde** (event-seviyesinde DEĞİL) 19 cycle zaten ≥1 SHORT sinyali taşıyor, **21 cycle hiç SHORT taşımıyor** (genuinely yeni SHORT-cycle temsili). Route overlap: SHORT_NEITHER=20, BUY_FADE_SHORT_H45_SL75=2, SHORT_NOISY_BTC1M_D5_H180=0. Data-quality: 54 pencerede 0 gerçek candle-gap/GAPPED-kalite satırı (görünen sayım farkları yalnız pencere-sınırı hizalama artefaktıydı). Aylık dağılım: 02:5/03:8/04:6/06:33/07:2 (Haziran'a aşırı yığılmış).
- Karar kuralı (operatör frozen): yeni-SHORT-cycle N<20→RETIRE; ≥20 ama split yetersiz→LIMITED_DESCRIPTIVE_POPULATION; yeterli→DISPOSABLE_CANONICALIZATION_REHEARSAL_READY. **21≥20 ama 16-hücre train/test split (MIN_BUCKET_N=20) için yetersiz → LIMITED_DESCRIPTIVE_POPULATION** (bu, disposable rehearsal'ın kendisini ENGELLEMEDİ — rehearsal'ın amacı chronological-split testi değil, identity/known-at/idempotency/rollback doğrulamasıydı).

### Disposable canonicalization rehearsal (bu batch'in kod üreten kısmı)
Yeni setup_id=**`SHORT_NOISY_BTC200K_CONFIRMED_V1`** (route_version=setup_id, eski `SHORT_NOISY_BTC1M_D5_H180`'dan kasıtlı FARKLI). Yeni dosyalar: `ami/lifecycle/short_noisy_v1_rehearsal.py` (identity/`build_signal_rows`/`backfill_short_noisy_v1`/field-provenance override/`rollback_short_noisy_v1` — `generate_signal_id`/`insert_transition` mevcut sözleşmeden BİREBİR reuse, `derive_signals()` KASITLI reuse edilmedi çünkü o anchor_ts_ms'i signal_birth_ts sayıyor) + `ami/lifecycle/short_noisy_v1_migration_rehearsal.py` (disposable-copy harness, `path_migration_rehearsal.py` deseniyle: baseline-fingerprint→disposable-copy→identity-determinism-check→backfill→field-provenance→path-field-provenance→path-metrics(freeze_and_record mevcut halinde, DEĞİŞTİRİLMEDEN, 324=270+54 sinyal üzerinde)→idempotency-rerun→old-reader-compat→overlap-matrix→rollback→reapply→schema-fingerprint-unchanged).

**Field-provenance dürüstlüğü:** 16 lifecycle-alanının 13'ü mevcut `FIELD_PROVENANCE_SPECS`'ten AYNEN reuse edildi. 3 alan (`signal_birth_ts`/`setup_id`/`route_version`) metni DÜZELTİLDİ çünkü bu setup için gerçek türetimleri farklı (conf_ts liquidation-scan / frozen literal / setup_id'ye eşit, `ami_events.route_version` comma-split DEĞİL) — `field_classification` hiçbirinde DEĞİŞMEDİ (hepsi DETERMINISTIC_HISTORICAL_SAFE kaldı, proxy↔safe geçişi YOK, testte assert edildi). 23 path-observation alanı hiç değiştirilmeden reuse edildi (path metrikleri signal_birth_ts'in türetim yönteminden bağımsız).

**Gerçek (disposable) sonuç:**
- candidate_n=54=source_event_n (1 sinyal/event), 40 distinct cycle, 19 zaten-SHORT/21 yeni-SHORT (cycle-seviyesi).
- Zorunlu kontroller BİREBİR geçti: identity 2. çağrıda deterministic; tüm conf_ts>noisy_ts+5m VE tüm noisy_ts>anchor+1m (no-lookahead); tüm signal_birth_ts==conf_ts; 2 duplicate conf_ts farklı event'leri BİRLEŞTİRMEDİ (54 farklı signal_id); 4-katman idempotency (signal/transition/lifecycle-provenance/path-provenance/path-metrics); old-reader-compat (mevcut 270/1080/10530 sayıları BİREBİR sabit); rollback (yalnız 54 signal/54 transition/216 path-obs/864+1242 provenance silindi); reapply byte-identical; schema fingerprint DEĞİŞMEDİ (bu batch'te 0 DDL).
- Path-metrikleri (324 sinyal üzerinde): +216 path-obs satırı; observation_status OK 914→1096(+182)/MISSING_INTERNAL_GAP 153→187(+34)/EXCLUDED_NO_HORIZON_DATA 13→13(+0); volatility_status OK 912→1094(+182)/NOT_APPLICABLE 166→200(+34)/INVALID 2→2(+0).

### Verdict
**CANONICAL_MIGRATION_READY_LIMITED_DESCRIPTIVE_POPULATION** — identity/known-at/idempotency/rollback/old-reader-compat hiçbir blocker bulmadı, ama 21 yeni-SHORT-cycle chronological TRAIN/TEST split'i (MIN_BUCKET_N=20) için yetersiz kalıyor; yalnız full-popülasyon tanımlayıcı analiz desteklenir. **Bu batch GERÇEK canonical.sqlite'a HİÇBİR YAZMA yapmadı** — yalnız disposable kopya üzerinde çalıştı (test içinde hash/mtime/schema-fingerprint BİREBİR sabit doğrulandı). Alfa/ekonomik iddia YOK, canonicalization için ayrı bir operatör onayı gerekir.

**Full regresyon:** yeni 9 test + yan-yana `test_ami_lifecycle_path_migration_rehearsal.py` (7/7) = 16/16 ✓, 0 regresyon. **Toplam AMI 630/630 ✓** (621 önceki + 9 yeni). Protected session delta ZERO.

WAIT_FOR_OPERATOR_APPROVAL

## 74. SHORT_NOISY_BTC200K_CONFIRMED_V1 — CONTROLLED CANONICAL DATA BACKFILL, APPLIED (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE SHORT_NOISY_BTC200K_CONFIRMED_V1 CONTROLLED CANONICAL DATA BACKFILL" — §73'ün disposable rehearsal'ının kabulünün devamı. **Identity/path/provenance backfill ONLY — alfa promosyonu, outcome test, ekonomik iddia veya observer aktivasyonu DEĞİL.**

### Preflight (birebir doğrulandı, 0 blocker)
schema_version=10, events=252, signals=270 (LONG=220/SHORT=50), raw_transitions=802, effective_transitions=270, path=1080, provenance=10530, FORWARD_N=0 (processed_trades=0, FORWARD_OBSERVATION sinyal=0), bindings=2 — operatörün beklediği checkpoint'le tam eşleşti. Aktif yazıcı yok (`BEGIN IMMEDIATE` başarılı). Backup: `data/ami/backups/canonical_pre_short_noisy_v1_backfill_20260704_203444.sqlite`, sha256=`bdb8cf88...` (kaynakla birebir doğrulandı).

### Uygulanan yazmalar (operatör-onaylı liste, fazlası yok)
`backfill_short_noisy_v1()` → 54 sinyal + 54 SIGNAL_BIRTH transition; `backfill_short_noisy_v1_field_provenance()` → 864 lifecycle-provenance satırı; `backfill_path_field_provenance()` → 1242 path-provenance satırı; `freeze_and_record()` (path_metrics, değiştirilmeden, artık 324 sinyal üzerinde) → 216 yeni path-obs satırı. Yeni `ami_events` YOK, mevcut hiçbir sinyal kimliği değişmedi.

### Postflight (operatörün beklediği tam sayılarla birebir)
events=252 (değişmedi), signals=324, LONG=220 (değişmedi), SHORT=104, raw_transitions=856, effective_transitions=324, current_state=324 OPEN, effective_CLOSED=0, path=1296, provenance=12636, independent-cycle evreni=167 (değişmedi), schema_version=10 (değişmedi), schema fingerprint DEĞİŞMEDİ (0 DDL), bindings=2 (değişmedi). observation_status OK=1096/MISSING_INTERNAL_GAP=187/EXCLUDED_NO_HORIZON_DATA=13; volatility_status OK=1094/NOT_APPLICABLE=200/INVALID=2 — beklenen dağılımlarla birebir. **Mevcut 270 sinyal/1080 path-obs/10530 provenance satırı content-hash olarak BYTE-IDENTICAL kaldı** (tüm-tablo ve SETUP_ID-hariç-scope'lu, her ikisi de doğrulandı).

### Zorunlu assertion'lar — hepsi PASSED
Tüm signal_birth_ts==conf_ts; tüm confirmation frozen noisy-window'dan strictly sonra; hiçbir identity alanı post-birth bilgiye bağlı değil; 54 signal_id hepsi distinct; duplicate conf_ts (2 çift) farklı event'leri birleştirmedi; direction hâlâ HISTORICAL_PROXY (proxy→safe upgrade yok); terminal_ts/first_executable_ts=NULL; OPEN="terminal doğrulanmadı", asla "aktif/live" değil. İdempotency: real DB'ye 2. kez uygulandı, sayı/content-hash birebir aynı (deterministic no-op). Rollback rehearsal: backfill-sonrası bir disposable kopyada `rollback_short_noisy_v1()` çalıştırıldı, yalnız 54/54/216/864+1242 silindi, sonuç tam 270/802/1080/10530'a döndü (gerçek DB'ye rollback UYGULANMADI, yalnız mekanizma doğrulandı).

### Full regresyon ve checkpoint güncellemeleri
48 AMI dosyası + 2 buyfade-mutation dosyası (630 test) çalıştırıldı. 8 önceden-yeşil dosyada (`test_ami_lifecycle_migration_rehearsal`/`path_field_provenance`/`path_metrics`/`path_migration_rehearsal`/`provenance_rehearsal`/`research_feature_gateway`/`research_w8_vol_normalized_baseline`) eski hardcoded 270/802/1080/4320/266 checkpoint'leri 324/856/1296/5184/324'e güncellendi — test LOGIC'i zayıflatılmadı, yalnız stale sayılar düzeltildi (emsal: P7A-P-CANON/P7B-CANON'un aynı disiplini). `migration_rehearsal.py`'ye yeni `reapply_signal_count`/`pre_rollback_signal_count` alanları eklendi (reapply-sonrası sinyal sayısının da — transition sayısı gibi — DOĞAL olarak farklı olduğu belgelendi: `backfill_lifecycle()` yalnız route_version-token popülasyonunu[270] yeniden kurar, SHORT_NOISY_V1'in ayrı kimlik kaynağından habersiz).

**ÖNEMLİ BULGU (yorumlanmadı, ayrı onaya bırakıldı):** SHORT popülasyonu büyüyünce (265→316) W8-VOL-NORMALIZED-BASELINE'ın 6/8 SHORT hücresi (5 `ANSWERED_REGIME_DEPENDENT_BASELINE` + 1 `ANSWERED_SUPPORTED_STABLE_BASELINE`) `INSUFFICIENT_SAMPLE` eşiğini ilk kez aştı (yalnız swing_24h/{mfe,mae} hâlâ yetersiz). Bu batch'te NO_OUTCOME_ANALYSIS/NO_TRAIN_TEST_CLAIM gereği bu sonuç YORUMLANMADI — ilgili test artık spesifik bir sınıflama iddia etmiyor (yalnız makinenin çalıştığını doğruluyor). **Bu, ayrı, açıkça onaylı bir W8 re-run/re-preregistration gerektiren YENİ bir açık soru olarak flag'lendi** (population büyümesi = yeni preregistration tetikleyicisi, mevcut donmuş sonuca sessizce eklenmemeli).

Regresyon sonunda **630/630 ✓**, gerçek DB hash bu regresyon boyunca BİREBİR SABİT kaldı (conftest.py'nin session-teardown assertion'ı hiç tetiklenmedi). `data/microstructure.db` yalnız mode=ro açıldı, hiç değişmedi.

MIGRATION_LOG.md'ye entry EKLENMEDİ — repository policy ("Yalnız schema/migration değişikliğinde güncellenir") bu batch'i kapsamıyor (0 DDL, salt veri backfill'i; emsal: W8-HOLD-BASELINE/W8-VOL-NORMALIZED-BASELINE'ın experiment_registry/results yazmaları da MIGRATION_LOG'a girmedi).

### Verdict
**Canonical-backfill closure: APPLIED_CLEAN, 0 blocker.**
**Descriptive-research readiness: CANONICAL_MIGRATION_READY_LIMITED_DESCRIPTIVE_POPULATION (değişmedi) + W8-SHORT-SUFFICIENCY-REOPENED (yeni, ayrı onay gerektiren açık soru — SHORT artık 6/8 W8 hücresinde MIN_BUCKET_N'i geçiyor, ama bu batch'te hiçbir outcome/train-test iddiası yapılmadı).**

WAIT_FOR_OPERATOR_APPROVAL

## 75. W8 EXPANDED-SHORT BASELINE — PREREGISTRATION + CYCLE-GROUPED VERSIONED RERUN, APPLIED (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE W8 EXPANDED-SHORT BASELINE PREREGISTRATION AND VERSIONED RERUN" — §74'ün incidental smoke-check bulgusunun (6/8 SHORT hücre INSUFFICIENT_SAMPLE eşiğini geçti) "scientific result olarak TREATED edilmemesi" gereğinin doğru şekilde ele alınması.

### Yeni deney kimlikleri (eskiler asla overwrite edilmedi)
`E-W8-HOLD-BASELINE-002-SHORT-EXPANDED`, `E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED` — `E-W8-HOLD-BASELINE-001`/`E-W8-VOL-NORMALIZED-BASELINE-001`'e yapısal olarak dokunulamaz (farklı `experiment_id` scope'u), `assert_old_experiments_untouched()` ile snapshot-hash öncesi=sonrası byte-identical doğrulandı.

### Mandatory cycle-grouped split
Yeni `ami/research/w8_short_expanded_baseline.py`: `compute_global_cycle_split()` — TÜM SHORT satırları (herhangi ufuk) independent_cycle_id'ye gruplandı, her cycle'ın EN ERKEN signal_birth_ts'i (üye satırları arasında, EN GEÇ değil) sıralama anahtarı, cycle-sayısına göre 70/30 kesildi. TEK global partition (`train_cycle_keys`/`test_cycle_keys`) 16 hücrenin TAMAMINDA aynen reuse edildi — bir cycle asla bir hücrede TRAIN diğerinde TEST olamaz (`assert_zero_cycle_straddling()`, her hücrede 0 ihlal doğrulandı). **MIN_BUCKET_N=20 artık independent-cycle N'e uygulanıyor**, signal-level N'e değil.

### Primary family (16 SHORT-only hücre, TEK ortak Holm)
4 metrik (mfe_bps/mae_bps/mfe_anchor_vol_units/mae_anchor_vol_units) × 4 ufuk (scalp_30m/scalp_1h/swing_4h/swing_24h). 16 p-value BİRLİKTE Holm-düzeltildi (iki ayrı 8-hücreli düzeltme değil); depolama iki deney-id'sine bölündü ama çıkarım TEK bir aile.

### Gerçek sonuç
60 distinct SHORT independent cycle (tüm ufuklar) → global split **42 TRAIN / 18 TEST cycle, 0 straddle**. Ufuk-bazlı: scalp_30m 42/18, scalp_1h 41/18, swing_4h 37/16, swing_24h 23/9. **TEST tarafı hiçbir ufukta 20'ye ulaşmıyor → 16/16 hücre INSUFFICIENT_SAMPLE** (bootstrap/permütasyon/Holm hiçbirinde hesaplanmadı, hepsi None, dürüstçe). Bu, §74'ün "6/8 yeterli" smoke-check bulgusunun signal-level split'in bir ARTEFAKTI olduğunu kanıtlıyor — cycle-grouped disiplin altında gerçek sonuç tam tersi.

**Final verdict: `EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT`.**

### Composition diagnostic (ikincil, tanımlayıcı-only, p-value YOK)
Pre-existing SHORT vs yeni SHORT_NOISY_BTC200K_CONFIRMED_V1 vs combined, ufuk başına N/median raporlandı (örn. scalp_30m: pre-existing mae_bps medyan=-17.36, yeni=-32.78, combined=-21.56). Üç dilim BAĞIMSIZ doğrulama olarak sunulmadı — yalnız descriptive context.

### Bütünlük
Gerçek DB'ye yazılan: yalnız `experiment_registry`(2 yeni satır)+`experiment_results`(15+15=30 satır). `ami_events`/`ami_signal_lifecycle`/`ami_lifecycle_transitions`/`ami_lifecycle_path_observations`/`ami_lifecycle_field_provenance` (252/324/856/1296/12636) DEĞİŞMEDİ (doğrulandı). İdempotent rerun doğrulandı (satır sayıları 2. çalıştırmada aynı kaldı). Full regresyon: 641/641 ✓ (630 önceki + 11 yeni, tüm 51 AMI dosyası ikişerli+bazı tekli). Protected session delta ZERO.

**0 blocker.**

WAIT_FOR_OPERATOR_APPROVAL

## 76. W8-LONG-TIMING-STRUCTURE-001 — WHEN LONG'S MFE/MAE OCCUR, APPLIED (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE W8-LONG TIMING-STRUCTURE-001" — §75'in `EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT` sonucu kabul edildi (MIN_BUCKET_N zayıflatılmadı, ufuklar birleştirilmedi, yönler pool'lanmadı, signal-level split'e dönülmedi — SHORT accumulation/new-event-family branch olarak kalıyor); READY olan sıradaki LONG dalgasına geçildi.

### Araştırma soruları ve yanıtları
1. **MFE mi MAE'den önce mi oluşuyor?** `intrabar_order_status` dağılımı: scalp_30m'de dengeli (MFE_FIRST %48.1/MAE_FIRST %51.9), ufuk uzadıkça MAE_FIRST baskınlaşıyor (scalp_1h %44.8/%55.2, swing_4h %34.9/%65.1, swing_24h %30.1/%69.9). 0 SAME_CANDLE_UNKNOWN, 0 ZERO_AT_REFERENCE (her iki time_to=0) bu popülasyonda gözlendi.
2. **Ne kadar sürede oluşuyor?** time_to_mfe_ms/time_to_mae_ms + fraction-of-horizon kantilleri ufuk başına raporlandı (deney kaydında tam liste).
3. **Kronolojik olarak stabil mi?** PRIMARY FAMILY (8 hücre: 2 timing metrik × 4 ufuk, LONG-only) — cycle-grouped 70/30 split, TEK ortak Holm. **8/8 hücre YETERLİ ve `ANSWERED_SUPPORTED_STABLE_BASELINE`** (yalnız swing_24h/time_to_mfe_ms pre-Holm nominal p=0.015 taşıyor, ortak Holm sonrası p=0.12, anlamsız).
4. **4 ufuk arasında nasıl değişiyor?** Her istatistik ayrı ayrı raporlandı — MFE_FIRST oranı ufuk uzadıkça azalıyor (yorumlanmadı, yalnız gözlemlendi).

### Split disiplini
Yeni `ami/research/w8_long_timing_structure.py`, `ami/research/w8_short_expanded_baseline.py`'nin cycle-grouped split makinesini (`compute_global_cycle_split`/`split_rows_by_cycle_keys`/`assert_zero_cycle_straddling`/`_cycle_key`) BİREBİR reuse etti (test'te `is` identity ile doğrulandı — bu fonksiyonlar SHORT'a özgü değildi). 216 LONG sinyal/214 source-event/**142 distinct independent cycle** → global split **99 TRAIN/43 TEST cycle, 0 straddle**. Ufuk-bazlı TEST-tarafı: scalp_30m 43, scalp_1h 42, swing_4h 38, swing_24h 30 — hepsi MIN_BUCKET_N=20'nin rahatça üzerinde (SHORT'un 60-cycle popülasyonundan çok daha geniş bir taban).

### Final verdict
**`LONG_TIMING_STRUCTURE_STABLE`.**

### Bütünlük
Gerçek DB'ye yazılan: yalnız `experiment_registry`(1 yeni satır: `E-W8-LONG-TIMING-STRUCTURE-001`)+`experiment_results`(15 satır). Tüm önceki deneyler (`E-W8-HOLD-BASELINE-001`/`E-W8-VOL-NORMALIZED-BASELINE-001`/iki `-002-SHORT-EXPANDED`) hiç dokunulmadı (farklı experiment_id scope'u). `ami_events`/`ami_signal_lifecycle`/`ami_lifecycle_transitions`/`ami_lifecycle_path_observations`/`ami_lifecycle_field_provenance` (252/324/856/1296/12636) DEĞİŞMEDİ. İdempotent rerun doğrulandı. Full regresyon: **652/652 ✓** (641 önceki + 11 yeni, tüm 52 AMI dosyası ikişerli+bazı tekli). Protected session delta ZERO.

**0 blocker.**

WAIT_FOR_OPERATOR_APPROVAL

## 77. W8-LONG-NESTED-PATH-ACCUMULATION-001 — HOW MUCH ADDITIONAL MFE/|MAE| ACCUMULATES AS HORIZON EXPANDS, APPLIED (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE W8-LONG-NESTED-PATH-ACCUMULATION-001" — §76'nın `LONG_TIMING_STRUCTURE_STABLE` sonucu kabul edildikten sonra sıradaki descriptive LONG path-structure dalgası.

### Common-cohort zorunluluğu
Yalnız **4 ufukta BİRDEN** (`scalp_30m`/`scalp_1h`/`swing_4h`/`swing_24h`) `observation_status='OK'` olan LONG sinyaller — `fetch_common_cohort()` popülasyon kompozisyonunun ufuklar arasında DEĞİŞMEDİĞİNİ garantiliyor (her aralık aynı sabit sinyal kümesi üzerinden hesaplanıyor). 216 LONG-≥1-OK-ufuk sinyalden **93'ü incomplete-horizon olarak hariç tutuldu**, **123 kaldı** (123 source-event, 86 distinct independent cycle). Global cycle-grouped split: **60 TRAIN/26 TEST cycle, 0 straddle**.

### Nested non-negativity
`ami.lifecycle.path_metrics`'in `reference_price`/`effective_path_start_ts`'i ufuktan BAĞIMSIZ (yalnız `signal_birth_ts`'e bağlı) olduğu için ufuk pencereleri iç içe geçmiş (30m⊂1h⊂4h⊂24h) — MFE (running max) ve |MAE| matematiksel olarak yalnız artabilir. Bu VARSAYILMADI: `assert_nested_nonnegativity()` gerçek popülasyona karşı (EPSILON=1e-6 float-tolerance ile) çalıştırıldı → **0 ihlal**.

### Primary family (6 hücre, tek ortak Holm)
delta_mfe_30m_to_1h/1h_to_4h/4h_to_24h + delta_abs_mae_30m_to_1h/1h_to_4h/4h_to_24h. **6/6 hücre YETERLİ ve `ANSWERED_SUPPORTED_STABLE_BASELINE`** — yalnız `delta_mfe_4h_to_24h` pre-Holm nominal p=0.0385 taşıyor, ortak 6-hücreli Holm sonrası p=0.231 (anlamsız).

### Final verdict
**`LONG_NESTED_PATH_STABLE`.**

### Secondary descriptive bulgu (p-value YOK, hold/exit önerisi DEĞİL)
Medyan artımlı |MAE| **HER ÜÇ aralıkta da 0.0 bps** — medyan sinyalin adverse excursion'ı ilk 30 dakikada tamamen gerçekleşmiş. Medyan artımlı MFE ise büyümeye devam ediyor (+48.76bps 1h→4h, +81.43bps 4h→24h). Medyan 24h-yakalama oranları: MFE %13.5/%19.5/%60.2 (30m/1h/4h), |MAE| %54.2/%81.9/%100.0 (30m/1h/4h). Bu, yalnız gözlemlenen path-davranışının bir tanımı — operatörün açık talimatı gereği ÖNERİLEN HOLD SÜRESİ olarak yorumlanmadı.

### Bütünlük
Gerçek DB'ye yazılan: yalnız `experiment_registry`(1 yeni satır: `E-W8-LONG-NESTED-PATH-ACCUMULATION-001`)+`experiment_results`(10 satır). Önceki 5 deneyin (`E-W8-HOLD-BASELINE-001`/`E-W8-VOL-NORMALIZED-BASELINE-001`/iki `-002-SHORT-EXPANDED`/`E-W8-LONG-TIMING-STRUCTURE-001`) hiçbirine dokunulmadı. `ami_events`/`ami_signal_lifecycle`/`ami_lifecycle_transitions`/`ami_lifecycle_path_observations`/`ami_lifecycle_field_provenance` (252/324/856/1296/12636) DEĞİŞMEDİ. İdempotent rerun doğrulandı. Full regresyon: **666/666 ✓** (652 önceki + 14 yeni, tüm 53 AMI dosyası ikişerli+bazı tekli). Protected session delta ZERO.

**0 blocker.**

WAIT_FOR_OPERATOR_APPROVAL

## 78. AMI HISTORICAL CANDLE GAP REMEDIATION — SOURCE AUDIT + DISPOSABLE REHEARSAL ONLY (2026-07-04, Sonnet 5)

Operatör onayı: "APPROVE AMI HISTORICAL CANDLE GAP REMEDIATION — SOURCE AUDIT AND DISPOSABLE REHEARSAL ONLY" — W8-LONG volatility-state audit'in April-gap bulgusunun (§ önceki bölüm) sistematik takibi. **NO_REAL_CANONICAL_WRITE — bu batch tamamen source-audit + disposable rehearsal.**

### Phase A — tam-tarih gap envanteri (yalnız Nisan değil)
ETHUSDT 1m, 2026-02-15→2026-07-03: expected=199180, present=173464, **missing=25716**, **208 distinct gap koşusu**. 0 duplicate open_ts_ms, 0 non-monotonic timestamp, 0 invalid OHLC ilişkisi (mevcut veri temiz — sorun yalnızca EKSİKLİK, bozulma değil). **Aylık dağılım: Şub=42, Mar=164, Nis=4239, MAY=15315(!), Haz=5938, Tem=18** — Mayıs, Nisan'dan çok daha büyük bir sorunmuş (en büyük tek koşu: 2026-06-02→06-05, 4739 dk = 3.3 gün); operatörün orijinal April-odaklı çerçevesi genişletilerek TÜM tarih incelendi.

### Phase B — authoritative source reconciliation
`agg_trades` (microstructure.db) 174M satır/tam aralık kapsıyor, ama **208/208 gap koşusunun HİÇBİRİNDE eşleşen satır YOK** — epoch-tam doğrulama (ilk ad-hoc spot-check `datetime.strptime().timestamp()` kullanarak makinenin UTC+3 yerel saatiyle YANLIŞ POZİTİF vermişti; düzeltilmiş epoch-integer kontrolü 0 satır buldu, bu doğru sonuç). **Sonuç: local candle-builder backfill bug DEĞİL — gerçek upstream/collector-seviyesi eksiklik.** Proje zaten onaylı kaynağa (Binance USDT-M **Futures**, `fapi.binance.com`, mevcut `binanceusdm` ccxt entegrasyonu — ASLA spot) canlı bağlantı test edildi, başarılı. **25716/25716 mum (208/208 pencere) %100 retrievable** — 0 duplicate, 0 yanlış-hizalı, 0 geçersiz-OHLC, 0 mevcut-satırla-çakışma.

### Phase C — sidecar/disposable rehearsal
Yeni `ami/chart/candle_gap_repair_rehearsal.py`: `CANDLE_DEFINITION_VERSION_EXTERNAL_REPAIR="candle-binance-fapi-repair-v1"` (mevcut `candle-agg_trades-v1`'den UNIQUE(symbol,timeframe,open_ts_ms,version) kısıtıyla yapısal olarak ayrık — çakışma İMKANSIZ). `validate_kline_row()` (1m-alignment/OHLC-geçerlilik/close_time reddi) + duplicate-batch reddi. Disposable sonuç: 25716 aday → 25716 kabul, 0 red, 0 conflict; idempotent rerun (manifest-hash birebir); rollback (25716 satır silindi, pre-repair manifest'e BİREBİR); reapply (post-repair manifest'e BİREBİR).

### Phase D — impact rehearsal (disposable before/after)
1296 (sinyal,ufuk) çiftinden **170'i etkilendi**: 149'u forward-path-window gap düzelmesi (observation_status/mfe_bps/mae_bps/timing), **21'i yeni tespit edilen bir mekanizma** — 60-candle GERİYE DÖNÜK `realized_vol_at_anchor` penceresi de repaired candle'lardan etkilenebiliyor (yalnız `*_anchor_vol_units` alanları değişiyor, observation_status/mfe_bps/mae_bps AYNI kalıyor — path_metrics.py'nin DEĞİŞTİRİLMEYEN, mevcut iki-eksenli tasarımının [observation_status vs volatility_status] doğru çalıştığının ek kanıtı). **1126/1296 çift byte-identical kanıtlandı** (index_before==index_after aynı sonucu üretiyor — "unaffected" iddiası varsayılmadı, doğrulandı). observation_status OK 1096→1245 (+149)/MISSING_INTERNAL_GAP 187→38. Aylık recovery: Şub=4/Mar=63/Nis=20/Haz=62.

**Coverage-only readiness değişiklikleri (outcome/p-value OKUNMADI):**
- **LONG 24h-complete common cohort: 123→194 sinyal** (86→131 cycle, TRAIN/TEST 60/26→91/40) — belirgin iyileşme, §77'nin April/right-censoring endişesini önemli ölçüde azaltıyor.
- Expanding-window vol-state HIGH: hâlâ INSUFFICIENT (test_cyc=19, DEĞİŞMEDİ).
- SHORT expanded baseline: hâlâ INSUFFICIENT (test_cyc 18→19, +1, verdict'i ÇEVİRMİYOR).
- rs_state: ETKİLENMEDİ (mark_prices'tan türetiliyor, ami_candles'tan değil).

### Verdict
**`HISTORICAL_CANDLE_REPAIR_READY_FOR_CONTROLLED_BACKFILL`.**

Tüm tamamlanmış deneyler (E-W8-HOLD-BASELINE-001, E-W8-VOL-NORMALIZED-BASELINE-001, iki `-002-SHORT-EXPANDED`, E-W8-LONG-TIMING-STRUCTURE-001, E-W8-LONG-NESTED-PATH-ACCUMULATION-001) DOKUNULMADI — bu batch hiçbirine yazmadı (yalnız disposable candle rehearsal). Gerçek `data/ami/canonical.sqlite` hash/mtime bu batch boyunca BİREBİR SABİT kaldı; gerçek `data/microstructure.db`'ye de 0 write (yalnız mode=ro sorgular + harici Binance API okuma çağrıları).

Full regresyon: **675/675 ✓** (666 önceki + 9 yeni, tüm 54 AMI dosyası) — 1 geçici/tekrarlanmayan hata (`test_ami_lifecycle_short_noisy_v1_rehearsal.py`'nin microstructure.db-dokunulmadı kontrolü; canlı, sürekli-toplayan bağımsız bir arka-plan prosesin dosya mtime'ını değiştirmesinden kaynaklandı, izole rerun'da 12/12 ✓ — regresyon DEĞİL). Protected session delta ZERO.

**0 blocker.**

WAIT_FOR_OPERATOR_APPROVAL

## 79. AMI HISTORICAL CANDLE REPAIR — CONTROLLED CANONICAL BACKFILL + VERSIONED PATH CORRECTION, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE AMI HISTORICAL CANDLE REPAIR CONTROLLED CANONICAL BACKFILL + VERSIONED PATH CORRECTION" — §78'in disposable rehearsal'ının kabulünün devamı, GERÇEK `data/ami/canonical.sqlite`'A uygulandı.

### Part 0 — Source package freeze
`data/ami/candle_repair_source_package/` (raw_klines.json 25716 satır, request_manifest.json 216 request + her biri raw/normalized SHA256, gap_manifest_pre_repair.json, package_manifest.json). package_sha256=`bb660b80...`, row_content_sha256=`bf1b95b6...`. **Binance'e hiç tekrar bağlanmadan** `build_candidate_rows()` ile 25716/25716 deterministic reconstruction KANITLANDI.

### Part 1 — Preflight + backup
Checkpoint birebir: schema=10/events=252/signals=324/transitions=856/path=1296/provenance=12636/FORWARD_N=0/bindings=2. Backup: `data/ami/backups/canonical_pre_candle_repair_backfill_20260705_000709.sqlite` (sha256 doğrulandı, path/provenance/experiment content-hash'leri ayrıca kaydedildi). microstructure.db için operatörün istediği gibi mtime YERİNE 6-pencereli bounded collector-aware spot-check invariant kullanıldı (tam-aralık aggregate 650GB+ ölçekte impractical).

### Part 2 — Candle backfill
25716 1m satırı (`candle_definition_version="candle-binance-fapi-repair-v1"`, eskiden yapısal olarak ayrık) gerçek DB'ye yazıldı — missing 1m=0. Yeni `rederive_5m_with_source_traceability()`: `derive_higher_timeframe()` DEĞİŞTİRİLMEDEN reuse edildi (5m için ayrı authoritative kaynak İCAT EDİLMEDİ), yalnız o fonksiyonun sabit-kodlu `candle_definition_version`'ı gerçek 1m-çocuk versiyonlarını yansıtacak şekilde düzeltildi (path_metrics.py'nin zaten kullandığı comma-join blend convention) — 5292 5m satır (5036 yeni + 256 blended-düzeltmeli), missing 5m=0.

### Part 3 — Path versioning contract
`path_metrics.freeze_and_record()`'ın kendi docstring'i path observations'ı zaten "re-derivable materialization, NOT append-only" olarak tanımlıyor — operatörün 3. tercihini (repo rebuildable-materialized-state tanımlıyorsa controlled replacement) karşıladı. Naif "tüm 1296'yı yeni versiyonla paralel kopyala" REDDEDİLDİ (çift-sayım riski — belgelendi). Yeni `ami/lifecycle/path_candle_repair_correction.py`: `path_definition_version="path-v2-candle-repair-r1"` — yalnız GERÇEKTEN değişen 170 çifte yeni satır (observation_id hash'i path_definition_version içerdiği için orijinal "path-v2" ile ASLA çakışmıyor), orijinal 1296 satır hiç dokunulmadı.

### Part 4 — Targeted correction
149 forward-window (GAP→OK) + 21 vol-only (yalnız `*_anchor_vol_units`) = **170 satır**. **1126 satır byte-identical kanıtlandı.** Effective dağılım: observation_status **OK=1245/MISSING_INTERNAL_GAP=38/EXCLUDED_NO_HORIZON_DATA=13** (operatörün beklediğiyle BİREBİR); volatility_status OK=1243/NOT_APPLICABLE=51/INVALID=2. Provenance DEĞİŞMEDİ (12636, signal-seviyesinde).

### Part 5 — Deney immutability
6 tamamlanmış deneyin (`E-W8-HOLD-BASELINE-001`/`E-W8-VOL-NORMALIZED-BASELINE-001`/iki `-002-SHORT-EXPANDED`/`E-W8-LONG-TIMING-STRUCTURE-001`/`E-W8-LONG-NESTED-PATH-ACCUMULATION-001`) content-hash'i backup'la BİREBİR — hiç dokunulmadı.

### Part 6 — Readiness (coverage-only)
LONG 24h-complete cohort **194 sinyal/131 cycle/91-40 TRAIN-TEST**; SHORT expanded baseline 61 cycle/42-19 (hâlâ INSUFFICIENT); vol-state HIGH hâlâ INSUFFICIENT (test_cyc=19); rs_state ETKİLENMEDİ.

### Part 7 — Idempotency + rollback
Gerçek DB'ye rerun sıfır-yeni-satır/byte-identical doğrulandı. Rollback yalnız DISPOSABLE bir kopyada rehearsal edildi (gerçek DB'ye ASLA uygulanmadı) — tam pre-repair state'e (173464/34801/1296) BİREBİR döndü, reapply post-repair state'i (199180/40093) yeniden üretti.

### Part 8 — Postflight
4 önceden-yeşil dosyada (`path_migration_rehearsal`, `short_noisy_v1_migration_rehearsal`/`short_noisy_v1_rehearsal`, `w8_hold_baseline`) stale-checkpoint + reopened-sufficiency düzeltmesi yapıldı (test logic zayıflatılmadı). SHORT/swing_24h hücresinin candle repair sonrası signal-level sufficiency eşiğini geçtiği bulundu — **yorumlanmadı, yalnız flag'lendi** (ayrı, preregistered bir rerun gerektirir). Schema fingerprint DEĞİŞMEDİ (0 DDL).

### Verdict
**`CANDLE_REPAIR_AND_PATH_CORRECTION_APPLIED_CLEAN`.**

Full regresyon: **685/685 ✓** (675 önceki + 10 yeni, tüm 55 AMI dosyası). microstructure.db'nin SQLite header/change-counter'ının canlı collector'ın her commit'inde değiştiği (ara sıra bounded-prefix-hash kontrolünü tetikleyen ama gerçek veri değişikliği OLMAYAN) davranışı gözlemlendi ve dokümante edildi — rerun'da her zaman temiz geçti. Protected session delta ZERO.

**0 blocker.**

## 80. W8-LONG-NESTED-PATH-ACCUMULATION-002-CANDLE-REPAIR — CORRECTED CANDLE-DATA RERUN, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE W8-LONG-NESTED-PATH-ACCUMULATION-002 CORRECTED CANDLE-DATA RERUN" — §79'daki candle repair kapanışının kabulünün devamı: v001'in (§77) 123 sinyal/86 cycle popülasyonu, repair sonrası genişleyen effective cohort'a karşı YENİ, immutable bir deney olarak yeniden koşuldu. v001 hiçbir şekilde MUTATE EDİLMEDİ.

### Part 0 — Effective path selection safety
`ami/lifecycle/path_candle_repair_correction.py`'ye `fetch_effective_path_observations()` (bir (signal,horizon) çifti için corrected satır varsa onu, yoksa orijinali seçer; `equals` filtresi seçim SONRASI uygulanır — 21 vol-only çiftin ikisinin de `observation_status='OK'` göstermesinden kaynaklanan çift-sayım tehlikesini önlemek için) ve `effective_path_selection_audit()` eklendi. Gerçek DB'ye karşı doğrulandı: `physical_row_count_by_version={'path-v2':1296,'path-v2-candle-repair-r1':170}`, `duplicate_physical_pair_n=170`, **`effective_row_count=1296`** (operatörün beklediğiyle birebir), **`duplicate_effective_pair_n=0`**, `corrected_rows_supersede_n=170`. `equals={"observation_status":"OK"}` filtresi tam **1245** satır döndürüyor (operatörün beklediğiyle birebir). **`BLOCKED_BY_EFFECTIVE_PATH_SELECTION` TETİKLENMEDİ.**

### Part 1 — Common cohort
Yeni `ami/research/w8_long_nested_path_accumulation_002_candle_repair.py` — v001'in TÜM generic makinesini (`compute_derived_fields`, `assert_nested_nonnegativity`, `compute_cell`, `compute_secondary_descriptive`, `ALL_DELTA_FIELDS`) ve `w8_short_expanded_baseline`'ın split makinesini BİREBİR reuse ediyor; yalnız `fetch_common_cohort()` effective selector'ı kullanacak şekilde override edildi. Cohort: **signal_n=194** (v001: 123, +71), **independent_cycle_n=131** (v001: 86, +45), excluded_incomplete_horizon_n=25 (v001'de 93'tü — repair, eksik-ufuk kapsamının çoğunu kapattı), cycle_straddling_violations=0. Aylık dağılım: Şub=24/Mar=55/Nis=24/Haz=91 (Temmuz yok — right-censored). Route kompozisyonu: **%100 LONG_SILENCE** (tek route).

### Part 2 — Split
Fresh cycle-grouped split (v001'in split'i REUSE edilmedi): **TRAIN=91 cycle / TEST=40 cycle** (v001: 60/26).

### Part 3 — Frozen metrics
v001'in 6 alanı (`delta_mfe_{30m_to_1h,1h_to_4h,4h_to_24h}`, `delta_abs_mae_{...}`) aynen kullanıldı — ekleme/çıkarma yok.

### Part 4 — Primary family (6 hücre)
Tüm 6 hücre **STABLE** (Holm p=1.0 hepsinde, CI sıfırı kapsıyor). Medyanlar: `delta_mfe_30m_to_1h=0.0`, `delta_mfe_1h_to_4h=+27.79bps`, `delta_mfe_4h_to_24h=+74.70bps`, `delta_abs_mae_*=0.0` (üç aralıkta da). Nested non-negativity: **0 ihlal**.

### Part 5 — Secondary descriptive
Capture-fraction ve incremental-MFE/|MAE| istatistikleri v001'in `compute_secondary_descriptive()` fonksiyonuyla (reuse, override edilmedi) hesaplandı — hold/exit/stop-loss olarak YORUMLANMADI, yalnız betimsel.

### Part 6 — v001 karşılaştırması
`compare_with_v001()`: v001 ve v002 **BAĞIMSIZ TEKRARLAR OLARAK DEĞİL** (cohort'ları örtüşüyor — v002, v001'in strict superset genişlemesi) bir tutarlılık kontrolü olarak karşılaştırıldı. 6 hücrenin **hiçbiri değişmedi** (`changed=False` hepsinde), 6 medyanın hepsi aynı işaret. **`comparison_conclusion="REPLICATED_ON_CORRECTED_EXPANDED_COHORT"`.**

### Part 7 — Integrity
v001'in (`E-W8-LONG-NESTED-PATH-ACCUMULATION-001`) 10 `experiment_results` satırı byte-identical (önce/sonra). Canonical `ami_signal_lifecycle`(324)/`ami_lifecycle_path_observations`(1466)/`ami_lifecycle_field_provenance`(12636)/`ami_candles`(239273) tabloları **HİÇ DEĞİŞMEDİ**. `experiment_registry` 16→17 (yeni satır, `supersedes_experiment_id=E-W8-LONG-NESTED-PATH-ACCUMULATION-001`), `experiment_results` 199→213 (+14). İdempotent rerun doğrulandı (14→14, aynı verdict). Protected paths (`tools/s34_state_machine_live_executor.py`/`.env`/`execution/`/`risk/`/`brain/`) dokunulmadı.

### Verdict
**`LONG_NESTED_PATH_STABLE_CORRECTED_DATA`.**

Full regresyon: **689/689 ✓** (tüm 56 AMI dosyası, +9 yeni test `test_ami_research_w8_long_nested_path_accumulation_002.py`). Not: önceki narrative toplam "685" idi; bu batch'te doğrudan `--collect-only` ile ölçülen otoriter sayı 689 (685 rakamı geçmiş batch'lerin kümülatif anlatım özetiydi, kesin dosya-bazlı audit değildi — 689 artık ground-truth). Protected session delta ZERO.

**0 blocker.**

## 81. AMI EFFECTIVE-PATH AND EXPERIMENT-IMMUTABILITY SAFETY HARDENING, APPLIED (2026-07-05, Sonnet 5)

Reconciliation raporunun (§80 sonrası) doğurduğu 4 sistemik riski kapatan **safety hardening batch** (research/outcome recomputation YOK — kod+test+dokümantasyon).

**GOAL A — fail-closed path selection:** `ami/research/feature_gateway.py::fetch_path_observations()`'a `effective: bool = False` parametresi + yeni `AmbiguousPathVersionError` eklendi. Eşleşen satırlar birden fazla `path_definition_version` içeriyorsa VE caller ne exact bir `path_definition_version` pin'lemiş ne de `effective=True` geçmişse → fail-closed (lexical "en yeni" tahmini YOK). `ami/lifecycle/path_candle_repair_correction.py`'nin iki iç çağrısı (`fetch_effective_path_observations`, `effective_path_selection_audit`) `effective=True` ile güncellendi — davranışları değişmedi (hâlâ tam **1296** effective satır, **0** duplicate — gerçek veriyle doğrulandı).

**GOAL B — immutable experiment write guard:** yeni `ami/warehouse/experiment_ledger.py` (`record_experiment_registry`/`record_experiment_results`, `ImmutableExperimentConflict`). Eski `ON CONFLICT(experiment_id) DO UPDATE` + kör `DELETE FROM experiment_results` deseni **6 modülün tamamında** bu iki fonksiyonla değiştirildi: yeni id→INSERT; aynı id+bit-bit aynı içerik→NOOP_IDENTICAL (yazma yok); aynı id+herhangi bir içerik farkı→`IMMUTABLE_EXPERIMENT_CONFLICT` (hiç yazmadan reddeder). Bookkeeping timestamp'ler (`preregistered_at/started_at/completed_at/created_ms/updated_ms`) içerik karşılaştırmasından hariç (rerun'lar arası doğal olarak farklı, bilimsel içerik değil). Mevcut kodda draft/non-completed bir yaşam döngüsü YOK — her satır yazıldığı an tamamlanmış kabul edilir.

**GOAL C — legacy module treatment (5 modül):** `w8_hold_baseline.py`/`w8_vol_normalized_baseline.py`/`w8_short_expanded_baseline.py`/`w8_long_timing_structure.py`/`w8_long_nested_path_accumulation.py`(v001) `fetch_path_observations()` çağrılarına `path_definition_version="path-v2"` pin eklendi (dondurulmuş, repair-öncesi popülasyonu byte-bir reprodüksiyon). **Beklenmedik gerçek bulgu:** `E-W8-HOLD-BASELINE-001`/`E-W8-VOL-NORMALIZED-BASELINE-001` path-pin'e RAĞMEN artık reprodüksiyon DEĞİL — bu ikisi hem LONG hem SHORT'u kapsıyor (setup_id filtresi yok), ve `BATCH-SHORT-NOISY-V1-CANON-BACKFILL` (54 yeni SHORT sinyali, -001'ler donduktan SONRA eklendi) `ami_signal_lifecycle`'ın kendisini 266→317'ye büyütmüş (candle repair'dan tamamen bağımsız bir drift). Yeni guard bunu doğru şekilde **`IMMUTABLE_EXPERIMENT_CONFLICT`** ile durduruyor (GOAL C'nin kendi "OR prevent direct execution... with a clear frozen experiment error" maddesi tam bunu karşılıyor) — production kod DEĞİŞTİRİLMEDİ, yalnız bu iki modülün testleri `freeze_and_record()`'ın artık fail-closed olması gerektiğini doğrulayacak şekilde güncellendi. Diğer 5 experiment_id (SHORT-EXPANDED×2 + LONG-TIMING-STRUCTURE + LONG-NESTED-PATH-ACCUMULATION v001/v002) drift YOK, hepsi NOOP_IDENTICAL ile byte-identical reprodüksiyon kanıtlandı.

**GOAL D — testler (+18 yeni test, toplam 707):** `test_ami_research_feature_gateway.py`'ye +5 (ambiguity fail-closed, explicit-version bypass, effective=True bypass, real-data 1296/0-duplicate, corrected-row-supersedes-only-in-effective), yeni `test_ami_warehouse_experiment_ledger.py` +8 (insert/noop/conflict×2, missing-column, schema-sync guard, no-ON-CONFLICT/no-bulk-DELETE structural guard), yeni `test_ami_effective_path_and_experiment_immutability_hardening.py` +3 (5 reprodüksiyonlu id byte-identical, 2 drift'li id fail-closed+byte-identical, forbidden-terms guard), `test_ami_research_w8_hold_baseline.py`/`w8_vol_normalized_baseline.py` her biri net +1 (eski "idempotent rerun" testi `compute_metrics()`-only + ayrı bir "fails-closed-on-real-drift" testine bölündü), `w8_short_expanded_baseline.py`'nin testi artık -001/-VOL-001'i fresh çağırmıyor (zaten var olan, artık reprodüklenemez satırları doğrudan okuyor).

**Doğrulamalar:** full regresyon **707/707 ✓** (57 dosya, ≤2-dosya/çağrı sequential, hepsi yeşil — `--collect-only` ile ölçülen ground-truth, 689+18). Gerçek `data/ami/canonical.sqlite`: mtime/schema_version(10)/events(252)/signals(324)/path_rows(1466)/experiment_registry(17) **HİÇ DEĞİŞMEDİ** (yalnız read-only + conftest'in izole disposable kopyası üzerinden test edildi — conftest'in kendi session-teardown assertion'ı da her 24 batch'te sıfır ihlal verdi). Protected paths dokunulmadı. NO_OUTCOME_ANALYSIS/NO_NEW_EXPERIMENT_RESULT/NO_P_VALUE — 6 W8 deneyinin hiçbirinin scientific_verdict/dataset_hash'i değişmedi (2 drift'li id için yazma zaten reddedildi, 5 reprodüksiyonlu id için NOOP_IDENTICAL).

**Verdict: `EFFECTIVE_PATH_AND_EXPERIMENT_IMMUTABILITY_HARDENED`.**

## 82. W8-SHORT-EXPANDED-BASELINE-003-CANDLE-REPAIR — CORRECTED CANDLE-DATA IMMUTABLE RERUN, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE W8 SHORT-EXPANDED BASELINE 003 CORRECTED CANDLE-DATA IMMUTABLE RERUN" — §81 hardening batch'inin kabulünün ardından ranked rerun queue'nun #1 maddesi (§80 reconciliation raporu).

**İki yeni immutable id:** `E-W8-HOLD-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR` / `E-W8-VOL-NORMALIZED-BASELINE-003-SHORT-EXPANDED-CANDLE-REPAIR` — `E-W8-HOLD-BASELINE-002-SHORT-EXPANDED`/`E-W8-VOL-NORMALIZED-BASELINE-002-SHORT-EXPANDED`'in corrected-data rerun'ı (`supersedes_experiment_id`), **-002 hiç çalıştırılmadı/mutate edilmedi**. Yeni `ami/research/w8_short_expanded_baseline_003_candle_repair.py`: -002'nin generic makinesinin (`compute_cell`/`compute_global_cycle_split`/`split_rows_by_cycle_keys`/`assert_zero_cycle_straddling`/`_cycle_key`/`_cell_rows`/`compute_composition_diagnostic`) TAMAMI `is`-identity ile reuse edildi; yalnız `fetch_raw_bps_population`/`fetch_vol_normalized_population` `fetch_effective_path_observations()`'a override edildi (v002 nested-path presedansı).

**Part 0 — mandatory effective-path integrity gate (BLOCKED_BY_EFFECTIVE_PATH_SELECTION mekanizması gerçekten implemente edildi):** `verify_effective_path_selection_integrity()` — gerçek veride physical_row_count_total=**1466** ✓, duplicate_physical_pair_n=**170** ✓, effective_row_count=**1296** ✓, duplicate_effective_pair_n=**0** ✓ — hepsi eşleşti, `passed=True`. Sentetik testlerle mismatch senaryosu da doğrulandı (`compute_family()` hiç popülasyon çekmeden erken `blocked=True`/`BLOCKED_BY_EFFECTIVE_PATH_SELECTION` döndürüyor).

**Population + split:** SHORT-only, effective selector; **global cycle split TAMAMEN operatörün "current coverage-only expectation"iyle birebir eşleşti: total_cycle_n=61, train_cycle_n=42, test_cycle_n=19** (zorlanmadı, gerçek popülasyondan hesaplandı — doğrulama, varsayım değil). Ufuk bazında: scalp_30m/scalp_1h cycle_n=61 (train42/test19), swing_4h cycle_n=59 (train41/test18), swing_24h cycle_n=49 (train38/test11) — **hepsi test_cycle_n<20**.

**Pre-outcome sufficiency gate:** MIN_BUCKET_N=20 DEĞİŞTİRİLMEDİ. Operatörün beklediği gibi **16/16 hücre `INSUFFICIENT_SAMPLE`** (bootstrap/permütasyon/Holm hiçbiri hesaplanmadı — `holm_adjust()` 16 p-value'nun hepsi None olduğu için n=0, gerçek işlem yapmadı). Her insufficient hücreye `descriptive_only_label="DESCRIPTIVE_ONLY_NOT_INFERENTIAL"` etiketi eklendi; full-population medyan/kantil/IQR alanları hâlâ raporlanıyor ama stable/regime-dependent/null/alpha iddiasına ASLA dönüştürülmedi. **Final verdict: `EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT_CORRECTED_DATA`.**

**Correction impact audit (bağımsız yeniden-türetildi, varsayılmadı):** SHORT'a özgü etkilenen fiziksel satır=**45**, distinct sinyal=**28**, event=**24**, cycle=**18**, class_b(vol-only)=**0** — reconciliation raporunun tüm beklentileriyle **birebir eşleşti** (0 mismatch), `swing_24h`'de yoğunlaşma (27/45) doğrulandı.

**Composition diagnostic + comparison-with-v002:** pre-existing/SHORT_NOISY_V1/combined ayrı raporlandı (bağımsız doğrulama olarak yorumlanmadı). v002 karşılaştırması: population/cycle/split değişimleri raporlandı, **hiçbir hücre INSUFFICIENT_SAMPLE'dan çıkmadı** (`any_cell_changed_from_insufficient_sample=False`), v002/v003 family_verdict'leri her ikisi de insufficient (yalnız `_CORRECTED_DATA` suffix'i farklı).

**Immutable ledger:** yeni `record_experiment_registry`/`record_experiment_results` kullanıldı (ON CONFLICT/DELETE YOK). İlk çalıştırma=INSERT, idempotent rerun=NOOP_IDENTICAL (satır sayısı değişmedi), -002'nin `experiment_results`'ı byte-identical kaldı (`old_experiments_untouched=True`).

**Testler:** yeni `test_ami_research_w8_short_expanded_baseline_003.py` — **+13 yeni test** (forbidden-terms static guard, yeni-id/`supersedes` kilidi, version-identifier kilidi, `is`-identity reuse kilidi [+ yalnız 2 fetch fonksiyonunun override edildiği kilidi], sentetik integrity-pass/fail + `compute_family` blocked-branch + `freeze_and_record` blocked-branch-write testleri, gerçek-veri correction-impact-audit/coverage-expectation/16-hücre-insufficient/Holm-no-op/descriptive-label/comparison-with-v002/idempotent-ledger testleri). **Toplam AMI 720/720 ✓** (58 dosya, `--collect-only` ground-truth, ≤2-dosya/çağrı sequential). Gerçek `data/ami/canonical.sqlite`: mtime/schema(10)/events(252)/signals(324)/path_rows(1466)/experiment_registry(17) **HİÇ DEĞİŞMEDİ**. Protected paths dokunulmadı. **0 blocker.**

**Verdict: `EXPANDED_SHORT_INSUFFICIENT_AFTER_CYCLE_GROUPED_SPLIT_CORRECTED_DATA`.** Protected delta ZERO.

## 83. W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED — RAW LONG BASELINE, CORRECTED DATA + CYCLE-GROUPED SPLIT, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE W8 LONG RAW HOLD BASELINE CORRECTED-DATA + CYCLE-GROUPED VERSION". Yeni tek immutable id: `E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED` — `E-W8-HOLD-BASELINE-001`'in LONG-only raw mfe_bps/mae_bps kısmının corrected-data + cycle-grouped-split rerun'ı (`supersedes_experiment_id=E-W8-HOLD-BASELINE-001`); **v001 hiç çalıştırılmadı**. **Bağımsız tekrar OLARAK TANIMLANMADI** — hem veri (effective/repaired path) hem split metodolojisi (cycle-grouped) değişti; `historical_reference_experiment_id`/`candle_data_version`/`path_data_version`/`methodological_change="SIGNAL_LEVEL_SPLIT_TO_INDEPENDENT_CYCLE_GROUPED_SPLIT"` explicit kaydedildi.

Yeni `ami/research/w8_hold_baseline_004_long_corrected_cycle_grouped.py`: cycle-grouped split makinesi (`compute_global_cycle_split`/`split_rows_by_cycle_keys`/`assert_zero_cycle_straddling`/`_cycle_key`/`compute_cell`) `w8_short_expanded_baseline`'dan `is`-identity reuse edildi (w8_long_timing_structure/w8_long_nested_path_accumulation presedansı). Popülasyon `fetch_effective_path_observations()` ile. **Yalnız 8 hücre** (mfe_bps+mae_bps × 4 ufuk, LONG-only) — vol-normalized metrikler bu batch'e DAHİL EDİLMEDİ (operatör talimatı, ayrı bir sonraki immutable deney gerektirecek).

**Part 0 — effective-path integrity:** 1466/170/1296/0 gerçek veride hepsi eşleşti (`passed=True`).

**Global cycle split:** total=142, train=99, test=43 — MIN_BUCKET_N=20'nin RAHATÇA üzerinde tüm ufuklarda (test_cycle_n: scalp_30m/1h=43, swing_4h=43, swing_24h=42) → **8/8 hücre YETERLİ** (SHORT'un aksine, LONG için gerçek bootstrap/permütasyon/Holm çalıştı).

**Pre-outcome coverage report:** ufuk başına raw/event/cycle N + aylık dağılım + setup kompozisyonu (dominant `LONG_SILENCE` — scalp_30m'de 216/219, swing_24h'de 194/194 %100) + `signals_sharing_independent_cycle` (45 multi-signal cycle, 122 sinyal) + `source_events_carrying_multiple_long_signals_n=1` — hepsi MFE/MAE okunmadan ÖNCE raporlandı.

**8/8 hücre `ANSWERED_SUPPORTED_STABLE_BASELINE`** (Holm-adjusted p=1.0 hepsinde, bootstrap CI sıfırı kapsıyor). **Final verdict: `LONG_RAW_HOLD_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED`.**

**Comparison with v001 (bağımsız tekrar DEĞİL, önemli düzeltme):** v001'in family-seviyesi popülasyon toplamları (`raw_signal_n_population` vb.) LONG+SHORT KOMBİNE (266) — LONG-only v004 (219) ile doğrudan karşılaştırılması yanlış-yönlendirici olurdu, bu yüzden BİLEREK yapılmadı. Bunun yerine v001'in KENDİ LONG hücrelerinden (`cell_mfe_bps|{horizon}|LONG`, direction-özgü) ufuk-bazlı popülasyon türetildi: swing_24h 123→194 (repair-driven +71, önceki batch'lerle birebir tutarlı), scalp_30m 216→219, scalp_1h 210→218, swing_4h 192→214. **0/8 hücre verdict değiştirdi** (v001'de de hepsi stable idi) → **comparison_label=`QUALITATIVELY_CONSISTENT_AFTER_CORRECTION_AND_CYCLE_GROUPING`** (zorlanmadı, ölçüldü).

**Testler:** yeni `test_ami_research_w8_hold_baseline_004.py` — **+10 yeni test** (forbidden-terms + no-vol-normalized + no-matched-control-reconstruction statik guardlar, yeni-id/methodoloji-kaydı kilidi, `is`-identity reuse kilidi, sentetik integrity-pass/fail + `compute_family`/`freeze_and_record` blocked-branch testleri, gerçek-veri smoke+idempotency+v001-byte-identical+8-hücre-MIN_BUCKET_N-disiplin+comparison-yapı kilitleri). **Toplam AMI 730/730 ✓** (59 dosya, `--collect-only` ground-truth).

**Gerçek `data/ami/canonical.sqlite`'A UYGULANDI:** `experiment_registry` 19→20 (+1), `experiment_results` 257→277 (+20); `ami_events`(252)/`ami_signal_lifecycle`(324)/`ami_lifecycle_path_observations`(1466)/`ami_lifecycle_field_provenance`(12636) HİÇ DEĞİŞMEDİ; v001 dataset_hash/results byte-identical kaldı; idempotent rerun aynı oturumda doğrulandı. **0 blocker.**

**Verdict: `LONG_RAW_HOLD_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED`.** Protected delta ZERO.

## 84. W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED — PAIRED VOL-NORMALIZED LONG BASELINE, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE W8 VOL-NORMALIZED BASELINE 004 LONG CORRECTED-DATA + CYCLE-GROUPED" — §83'ün kabulünün ardından eşleştirilmiş vol-normalized dalga. Yeni tek id `E-W8-VOL-NORMALIZED-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED` — `E-W8-VOL-NORMALIZED-BASELINE-001`'in LONG-only kısmının corrected-data+cycle-grouped rerun'ı (`supersedes_experiment_id=E-W8-VOL-NORMALIZED-BASELINE-001`), **v001 hiç çalıştırılmadı**. `historical_reference_experiment_id`/`paired_raw_baseline_experiment_id=E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED`/`candle_data_version`/`path_data_version`/`methodological_change` explicit kaydedildi. **DEĞİL** bir volatility-state stratification dalgası — HIGH/LOW etiket, medyan-eşik veya rejim-fitting hiçbiri YOK.

Yeni `ami/research/w8_vol_normalized_baseline_004_long_corrected_cycle_grouped.py`: cycle-grouped split makinesi `w8_short_expanded_baseline`'dan `is`-identity reuse; yalnız 8 hücre (mfe_anchor_vol_units+mae_anchor_vol_units×4 ufuk, LONG-only, raw-bps bu batch'e DAHİL EDİLMEDİ — o zaten eşleştirilmiş v004 raw deneyinde).

**MANDATORY SPLIT REUSE (yeniden-optimize edilmedi):** split, vol-filtrelenmiş popülasyondan DEĞİL, eşleştirilmiş raw deneyin (`w8_hold_baseline_004...fetch_population()`, doğrudan import — yeniden yazılmadı) kullandığı AYNI ham (observation_status=OK, LONG) popülasyondan yeniden hesaplandı — deterministik olduğu için byte-bir aynı sonucu üretir, ve bu VARSAYILMADI: `verify_split_matches_paired_raw_baseline()` v004'ün KENDİ depolanmış `global_cycle_split`'ini okuyup karşılaştırıyor. Gerçek veride **`matches=True`** (total=142/train=99/test=43, v004 ile birebir). Volatility_status nedeniyle dışlanan sinyaller (scalp_30m'de 1, scalp_1h'de 1, swing_4h/swing_24h'de 0) cycle'ın split tarafını hiç değiştirmedi (split zaten bu sinyalleri İÇEREN ham popülasyondan kuruldu).

**8/8 hücre YETERLİ, `ANSWERED_SUPPORTED_STABLE_BASELINE`** (Holm-p=1.0 hepsinde) — MIN_BUCKET_N=20 korunuyor, gerçek Holm/bootstrap/permütasyon çalıştı.

**Comparison with paired raw v004:** **0/8 hücre normalizasyon sonrası verdict değiştirdi** → `comparison_label=RAW_AND_VOL_NORMALIZED_LONG_BASELINES_CONSISTENT`, `raw_stability_conclusion_survives_normalization=True`.

**Comparison with historical v001:** (v004'ün kendi düzeltmesiyle aynı disiplin — v001'in family-seviyesi toplamları LONG+SHORT kombine, doğrudan karşılaştırılmadı; v001'in KENDİ LONG hücrelerinden ufuk-bazlı türetildi) **0/8 hücre değişti** → `comparison_label=QUALITATIVELY_CONSISTENT_AFTER_CORRECTION_AND_CYCLE_GROUPING`.

**Testler:** yeni `test_ami_research_w8_vol_normalized_baseline_004.py` — **+11 yeni test** (forbidden-terms + no-volatility-state-classification statik guardlar, yeni-id/metadata kilidi, `is`-identity reuse kilidi [split makinesi + `fetch_raw_population is w8h4.fetch_population`], sentetik integrity-pass/fail + `compute_family`/`freeze_and_record` blocked-branch testleri, gerçek-veri split-byte-exact-eşleşme kilidi, gerçek-veri smoke+idempotency+v001-VE-paired-raw-v004-byte-identical+8-hücre-MIN_BUCKET_N-disiplin+iki-comparison-yapı kilitleri). **Toplam AMI 741/741 ✓** (60 dosya, `--collect-only` ground-truth).

**Gerçek `data/ami/canonical.sqlite`'A UYGULANDI:** `experiment_registry` 20→21 (+1), `experiment_results` 277→300 (+23); `ami_events`(252)/`ami_signal_lifecycle`(324)/`ami_lifecycle_path_observations`(1466)/`ami_lifecycle_field_provenance`(12636) HİÇ DEĞİŞMEDİ; v001 VE eşleştirilmiş raw v004 byte-identical kaldı; idempotent rerun aynı oturumda doğrulandı. **0 blocker.**

**Verdict: `LONG_VOL_NORMALIZED_BASELINE_STABLE_CORRECTED_CYCLE_GROUPED`.** Protected delta ZERO.

## 85. W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED — CORRECTED CANDLE-DATA + EFFECTIVE-PATH RERUN, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE W8-LONG-TIMING-STRUCTURE-002 CORRECTED CANDLE-DATA + EFFECTIVE-PATH RERUN" — §83/§84'ün LONG corrected-data+cycle-grouped katmanının kabulünün ardından. Yeni tek id `E-W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED` — `E-W8-LONG-TIMING-STRUCTURE-001`'in corrected-data+cycle-grouped rerun'ı (`supersedes_experiment_id=E-W8-LONG-TIMING-STRUCTURE-001`), **v001 hiç çalıştırılmadı**. `corrected_data_rerun_of`/`candle_data_version`/`path_data_version`/`paired_cycle_split_experiment_id=E-W8-HOLD-BASELINE-004-LONG-CORRECTED-CYCLE-GROUPED` explicit kaydedildi. Hâlâ salt descriptive path-timing araştırması — entry/exit/stop/management/hold/ekonomik iddia YOK.

Yeni `ami/research/w8_long_timing_structure_002_candle_repair_cycle_grouped.py`: `fetch_population()` doğrudan `w8_hold_baseline_004...fetch_population()`'ı (`is`-identity, yeniden yazılmadı) kullanıp üstüne v001'in 3 frozen türetilmiş alanını (fraction-of-horizon×2, timing_delta_ms) ekliyor — bu, v002'nin timing popülasyonunun v004'ün KENDİ ham popülasyonuyla (gözlem-durumu=OK, LONG, tüm 4 ufuk) BİREBİR AYNI olduğu anlamına geliyor. `compute_cell`/`compute_horizon_descriptive`/`_rate`/`TIMING_METRICS` v001'den `is`-identity reuse edildi; split makinesi `w8_short_expanded_baseline`'dan.

**Mandatory split reuse:** split, v004'ün AYNI ham popülasyonundan yeniden hesaplandı — `verify_split_matches_paired_cycle_split()` v004'ün depolanmış `global_cycle_split`'iyle karşılaştırıp **`matches=True`** kanıtladı (total=142/train=99/test=43, v001'in KENDİ pre-repair split'iyle de aynı sayısal değer — açıklanabilir bir tesadüf: repair zaten-bilinen cycle'lara eksik UFUKLAR ekliyor, yeni cycle eklemiyor, bu yüzden cycle sayısı ve kronolojik sıralama değişmiyor).

**Correction impact audit (class_a-only, LONG):** **104 satır/71 sinyal/71 event/49 cycle — operatörün beklediğiyle BİREBİR eşleşti** (0 mismatch); 21 LONG class_b (vol-only) satırı bilinçli olarak HARİÇ tutuldu (timing alanlarını hiç değiştirmiyorlar). "Etkilenen" ve "yeni-uygun" aynı küme (gözlem-durumu=OK kapısı için GAP→OK geçişi ile eşdeğer, doğrulandı).

**8/8 hücre YETERLİ, `ANSWERED_SUPPORTED_STABLE_BASELINE`** (Holm-p=1.0 hepsinde).

**Comparison with v001:** **0/8 hücre değişti** → `comparison_label=TIMING_STRUCTURE_CONSISTENT_ON_CORRECTED_EXPANDED_COHORT`. Nitel bulgu ("ufuk uzadıkça MAE_FIRST oranı artıyor") **hayatta kaldı** (`mae_first_increases_with_horizon_survives=True`) — MAE_FIRST oranları scalp_30m→swing_24h: %52.1→%55.5→%63.1→%63.4 (v001: %51.9→%55.2→%65.1→%69.9, aynı monoton artış deseni korunuyor).

**Testler:** yeni `test_ami_research_w8_long_timing_structure_002.py` — **+10 yeni test** (forbidden-terms + no-short-pooling statik guardlar, yeni-id/metadata kilidi, `is`-identity reuse kilidi [split makinesi + v001'in timing-family fonksiyonları + `fetch_raw_population is w8h4.fetch_population`], sentetik integrity-fail + `compute_family`/`freeze_and_record` blocked-branch testleri, gerçek-veri correction-impact-audit (104/71/71/49 birebir) + split-byte-exact-eşleşme kilidi, gerçek-veri smoke+idempotency+v001-VE-paired-raw-v004-byte-identical+8-hücre-MIN_BUCKET_N-disiplin+secondary-descriptive-no-p-value+comparison-yapı kilitleri). **Toplam AMI 751/751 ✓** (61 dosya, `--collect-only` ground-truth).

**Gerçek `data/ami/canonical.sqlite`'A UYGULANDI:** `experiment_registry` 21→22 (+1), `experiment_results` 300→323 (+23); `ami_events`(252)/`ami_signal_lifecycle`(324)/`ami_lifecycle_path_observations`(1466)/`ami_lifecycle_field_provenance`(12636) HİÇ DEĞİŞMEDİ; v001 VE eşleştirilmiş raw v004 byte-identical kaldı; idempotent rerun aynı oturumda doğrulandı. **0 blocker.**

**Verdict: `LONG_TIMING_STRUCTURE_STABLE_CORRECTED_DATA`.** Protected delta ZERO.

## 86. AMI BIRTH-TRUNCATED CASCADE GEOMETRY — DISPOSABLE SCHEMA MIGRATION + CANONICAL BACKFILL REHEARSAL (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE AMI BIRTH-TRUNCATED CASCADE GEOMETRY DISPOSABLE SCHEMA MIGRATION + CANONICAL BACKFILL REHEARSAL ONLY" — bir önceki oturum kesintiye uğramıştı; bu batch Goals A-D'nin (schema/immutability/provenance/constraints, önceki oturumdan `ami/geometry/birth_truncated_cascade_geometry.py` + ilk test dosyası olarak diskte kalmış ama hiç pytest ile doğrulanmamıştı) doğrulanması + Goals E-I'nin (gerçek-veri disposable rehearsal, migration-safety, coverage-gate, verdict) tamamlanmasıdır. **REAL canonical.sqlite'a 0 write; microstructure.db'ye 0 write (mode=ro).**

**Goals A-D doğrulama:** önceki oturumdan kalan `ami/geometry/birth_truncated_cascade_geometry.py` (schema+backfill+immutability+append-only quality-ledger+field-provenance) ilk kez pytest ile çalıştırıldı — **2 gerçek bug bulundu ve düzeltildi** (kod DEĞİL, kalıntı test dosyasında): (1) 8 yerde `_liq(999_940_000, ...)` yanlış ölçekli zaman damgası (anchor_ts=1_000_000'a göre 60s önce olması gerekirken ~1000x büyük bir değerdi, bucket dışına düşürüyordu) → `_EARLY_TS = 1_000_000 - 60_000` sabitine düzeltildi; (2) append-only quality-assessment testinde ikinci (daha yeni) kayda `assessed_at_ms=999` verilmişti — ki bu, `backfill()`'in gerçek `time.time()*1000` ile damgaladığı ilk kayıttan SAYISAL OLARAK KÜÇÜK (yani "daha eski") kalıyordu, effective-view'ın "en yeni" seçimini yanlış test ediyordu → ilk kaydın gerçek `assessed_at_ms`'i okunup +1000 ile düzeltildi. **+1 yeni test** (`test_module_never_writes_to_ami_events`, Goal F: statik regex guard, modül kaynağında `ami_events`'e INSERT/UPDATE/DELETE yok).

**Goals E-I (yeni `ami/geometry/birth_truncated_geometry_rehearsal.py` + `tests/test_ami_geometry_birth_truncated_geometry_rehearsal.py`, +11 test):** gerçek `data/ami/canonical.sqlite`'ın disposable kopyasına + gerçek `data/microstructure.db`'ye (mode=ro) karşı tam rehearsal.

- **Population (ölçüldü, zorlanmadı):** 220 LONG sinyal / 218 kaynak-event / 142 bağımsız cycle / split TRAIN=99,TEST=43 — operatörün "confirmed findings"iyle **BİREBİR eşleşti**.
- **Backfill run 1:** candidate_n=220, accepted_n=220, rejected_n=0 (220/220 reconstructable — operatörün bulgusuyla eşleşti).
- **Data-quality partition — gap-registry mantığı bu oturumda YENİDEN inşa edildi** (önceki oturumun kesin algoritması diskte kalmamıştı, yalnız düzyazı ipucu vardı): cutoff = liquidations stream'i için TÜM gap satırlarının (resolved+unresolved) en son `start_ts_ms`'i (2026-04-27 14:27:24.680, gap id=783) — bunun ötesindeki hiçbir pencere `SOURCE_COMPLETE` kanıtlanamaz. Cutoff ÖNCESİ pencereler için "gapped" yalnız RESOLVED (start+end ikisi de bilinen) bir gap'le örtüşürse verilir — `ami.chart.candle_builder._load_agg_trades_gaps`'in AYNI emsali (açık/unresolved satırlar bounded "known gap" olarak sayılmaz). Ölçülen sonuç: **SOURCE_COMPLETE=125 / SOURCE_GAPPED=1 / SOURCE_COVERAGE_UNRESOLVED=94** (Şub+Mar+Nis=126 cutoff-öncesi → 125 complete+1 gapped; Haz+Tem=94 cutoff-sonrası → hepsi unresolved). **SOURCE_GAPPED=1 operatörün belirttiği "1"le BİREBİR eşleşti**; SOURCE_COMPLETE (125) ve SOURCE_COVERAGE_UNRESOLVED (94) operatörün önceki oturumdan aktardığı 83/136 rakamlarıyla EŞLEŞMEDİ (toplam 220 sabit, yalnız complete/unresolved sınırı farklı) — **açık madde**: önceki oturumun tam cutoff-seçim algoritması kodda hiç kalıcı olmamıştı, yalnız operatör promptunun düzyazısı vardı; bu oturumdaki yeniden-inşa gerekçeli ve mevcut repo emsaliyle (candle_builder) tutarlı ama orijinal oturumla byte-bir doğrulanamadı. **Operatör onayı gerekiyor**: hangi cutoff/gapped tanımı kanonik sayılacak.
- **per_feature_null_counts:** yalnız `inter_cluster_gap_sec`=1 NULL (ilk-anchor, tanımı gereği); diğer 7 alan hiçbir zaman NULL değil.
- **SOURCE_COMPLETE_ONLY popülasyonu (ölçülen 125 sinyal ile):** 87 cycle / TRAIN=60,TEST=27 — **MIN_BUCKET_N=20'yi HER İKİ split'te de geçiyor → `min_bucket_n_verdict=OK`** (yukarıdaki açık maddeye bağlı: cutoff tanımı değişirse bu rakam da değişir).
- **Migration-safety (Goal H) — hepsi YEŞİL:** idempotent rerun (row-count+content-hash birebir), conflicting-content aynı identity'de `ImmutableGeometryConflict` ile fail-closed, old-reader compatibility (ami_events/ami_signal_lifecycle/ami_lifecycle_path_observations/experiment_registry sayıları DEĞİŞMEDİ), rollback pre-migration schema-fingerprint'e BİREBİR döndü + mevcut satır sayılarını korudu, reapply content-hash'i BİREBİR yeniden üretti.
- **Semantic collision guard (Goal F):** modül dokümantasyonu + statik testler `ami_events.event_count`(route-count, liquidation-count DEĞİL)/`event_end_ts_ms`(post-birth, `feature_available_ts_ms` DEĞİL)/`notional`(farklı pencere) ayrımını kilitliyor; yeni tablo `ami_events`'i asla mutate etmiyor (regex-doğrulandı).

**Integrity:** gerçek `data/ami/canonical.sqlite` hash+mtime testte DEĞİŞMEDİ; `data/microstructure.db` boyutu yalnız büyüdü (canlı collector), ETHUSDT SELL likidasyon satır sayısı azalmadı (statik mode=ro guard + monotonic-size/count invariant — tam-dosya hash denenmedi, 650GB+ collector-aware emsal candle-repair batch'lerindeki gibi). Protected delta ZERO.

**Testler:** +32 yeni test (21 Goals-A-D + 11 Goals-E-I) toplamı bu iki yeni dosyada; her ikisi de **31/31 ✓** (schema dosyası 22 test — 21 orijinal +1 yeni Goal-F guard — ile rehearsal dosyası 11 test'in ikisi birlikte 33 test olarak, `--basetemp` scratchpad + `-p no:cacheprovider` + sequential 2-dosya kuralıyla çalıştırıldı). Tam `tests/` klasörü `--collect-only` ile toplandı (2611 test, `tests/legacy_tools/` içindeki 3 önceden-var basename-çakışması hatası HARİÇ — bu batch'le ilgisiz, dokunulmadı) — import/collection hatası YOK.

**Verdict (migration-readiness): `MIGRATION_READY_WITH_INFERENTIAL_SOURCE_QUALITY_BLOCKER`** — şema/backfill/immutability/rollback/reapply mekanik olarak tam kanıtlandı (gerçek canonical DB'ye UYGULANMADI, yalnız disposable), ama data-quality cutoff tanımının operatör onayı gerektiren bir açık maddesi var.
**Verdict (research-readiness): [§87 ile DÜZELTİLDİ] PROVISIONAL — `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`** — bu bölümün ilk halindeki `GEOMETRY_INFERENTIAL_RESEARCH_READY` ifadesi KABUL EDİLMEDİ (operatör talimatı); metodoloji uyuşmazlığı §87'deki reconciliation ile çözülene kadar hiçbir geometry outcome deneyi çalıştırılamaz, gerçek canonical migration yapılamaz.

Gerçek canonical.sqlite'a 0 write. **0 blocker (kod); 1 açık madde (data-quality cutoff tanımı onayı → §87).**

## 87. LIQUIDATION SOURCE-QUALITY COVERAGE CONTRACT RECONCILIATION — READ-ONLY (2026-07-05, Fable 5)

Operatör talimatı: §86'nın 83/1/136 (orijinal audit) vs 125/1/94 (rehearsal) uyuşmazlığı migration/deney öncesi çözülmeli; provisional verdict `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`. Tam rapor + 220 sinyallik per-signal tablo: `reports/research/s34/S34_LIQUIDATION_SOURCE_QUALITY_RECONCILIATION_2026-07-05.md` + `.json`; deterministic script: `tools/research_s34_source_quality_reconciliation.py`. READ-ONLY: canonical.sqlite sha256 önce/sonra BİREBİR (`c2b0b300…3098f`), microstructure.db mode=ro, outcome OKUNMADI, migration/deney YAZILMADI.

**GOAL A — iki yöntem de BİREBİR yeniden üretildi:** METHOD_A = resolved-gap overlap → GAPPED; sonra `birth >= İLK open-ended liq-gap start` (2026-04-02 17:58:38.989) → UNRESOLVED; kalan COMPLETE → **83/1/136 tam eşleşme**. METHOD_B = aynı ama cutoff = SON liq-gap-satır start (2026-04-27 14:27:24.680) → **125/1/94 tam eşleşme**. Uyuşmazlık kümesi **tam 42 sinyal, hepsi Nisan 2026** (birth ∈ (Apr 2, Apr 27)), 28 ayrık cycle, hepsi A=UNRESOLVED/B=COMPLETE. Tek GAPPED sinyal iki yöntemde de aynı (`SIG-291486…0109`, Apr 5, resolved gap id=114 overlap). **İkisi de pozitif kanıt kullanmıyor** — registry sessizliğinden tamlık çıkarıyorlar, yalnız sessizliğe nereye kadar güvendiklerinde ayrışıyorlar.

**GOAL B — registry semantiği (silinen writer kodu git stash `07e1a1f9`'dan kurtarıldı):** gap = >120s staleness heartbeat (satır `resolved_bool=0/end_ts_ms=NULL` açılır; veri dönünce `_resolve_gap` kapatır; >300s = critical/reconnect). **Resolution state in-memory idi** → her collector restart açık satırları sonsuza dek yetim bırakır. Güncel collector'da gap kodu HİÇ YOK (registry ölü; son satırlar: liq Apr 27 / agg Apr 24 / mark May 28). Registry İKİ YÖNDE DE güvenilmez KANITLANDI: (a) 21 open-ended satırın ampirik sessizliği ~0.1–2s (yanlış alarm/clock artifact'ı), (b) **gerçek kayıplar kaçırıldı** — liquidations tablosunda **40.1 günlük TAM blackout (2026-04-27 14:27:26 → 2026-06-06 17:43:52; Mayıs=0 satır, tüm semboller)** + Nisan'da saatlik delikler (Apr 24: 12.3h, Apr 27: 7.5h, Apr 23: 6.9h — yalnız sonuncusu resolved satır). **NO GAP RECORD ≠ SOURCE_COMPLETE, hiçbir dönemde** (METHOD_A ayrıca registry'nin var bile olmadığı Şub–Mar'a güveniyor; ilk registry satırı 2026-04-02).

**GOAL C — bağımsız kanıt:** (1) **Cross-stream health GEÇERSİZ:** Mayıs'ta aynı combined websocket 15.27M agg_trades + 260k mark_prices satırı taşırken liquidations 0 satır → haftalarca süren **liquidations-stream-only sessiz arıza kanıtlandı** (Apr 24 deliğinde de agg/mark aktı). (2) **Stream-mode keşfi:** Şub–Mar = 2 sembol, Nisan = 3 sembol (**per-symbol `@forceOrder`**); Jun 6+ = **733 sembol (`!forceOrder@arr` all-market**, güncel collector; veri 2026-06-06 17:43:52'de döndü). Per-symbol dönemde doğal sessizlikler dakikalar-saatler → pencere-bazlı cadence doğrulaması İMKANSIZ → **Şub–Nis sinyalleri için pozitif tamlık kanıtı YAPISAL olarak erişilemez**. All-market dönemde cadence yoğun/sağlıklı (1.13M satır, ayda 27 delik ≥120s, max 818s) → pencere-bazlı pozitif doğrulama MÜMKÜN.

**GOAL D — önerilen fail-closed contract (`liq-source-quality-contract-v2`):** (1) resolved registry gap overlap → GAPPED; (2) `birth < 2026-06-06 17:43:52` → UNRESOLVED (per-symbol dönem: kanıtlanamaz); (3) all-market dönemde `[ws−1800s, birth]` üzerinde max all-market inter-arrival ≤300s (collector'ın KENDİ frozen critical sabiti; 1800s = frozen MIN_GAP_SEC) → COMPLETE, değilse UNRESOLVED. Eşikler önceden-var frozen sabitler, hiçbir popülasyona/outcome'a fit edilmedi. Operatörün talimatıyla tutarlı: 42 ihtilaflı sinyal pozitif kanıtlanamıyor → UNRESOLVED.

**GOAL E — araştırma hazırlığı (dört aday):** METHOD_A: 83 COMPLETE → 59 cycle (41/18) **FAIL**; METHOD_B: 125 → 87 (60/27) geçer ama absence-of-evidence temelli; STANDARD-2 (cross-stream): 204 → 133 (93/40) geçer ama Mayıs kanıtıyla **GEÇERSİZ**; **CONTRACT-V2: 93 COMPLETE (Haz 90/Tem 3) → 54 cycle (37/17) FAIL**. Savunulabilir iki aday da (A ve V2) MIN_BUCKET_N=20'yi geçemiyor; geçen adaylar geçersiz akıl yürütmeye dayanıyor. **Research-readiness: `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`** (operatör contract'ı dondurunca V2 altında sonuç `GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY` olur). Doğru flag'lenmiş satırların canonical depolanmasını bloklamaz.

**GOAL F — test-ground-truth reconciliation:** **751/751** = `pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py --collect-only` (61 dosya = 59 AMI [711] + 2 buyfade mutation [24+16]) — BİREBİR yeniden üretildi; aynı küme bugün **783 = 751 + 32** yeni geometry testi topluyor. **2611** = `pytest tests/ --ignore=tests/legacy_tools --collect-only` (çok daha geniş kapsam — hiçbir zaman AMI regression ground truth olmadı; parametrizasyon değil KAPSAM farkı). **3 hata** = yalnız `tests/legacy_tools/` içinde collection error (legacy test dosyasının KENDİ içinde IndentationError + `tests/execution/` ile duplicate-basename import mismatch) — önceden-var, yapısal, AMI ile ilgisiz. **Önerilen frozen komut** (henüz kabul edilmedi — önce bir tam yeşil koşu şart): 751-set + 2 geometry dosyası → beklenen **783**; o zamana dek ground truth **751/751 kalır**.

**GOAL G:** §86'nın research-readiness satırı düzeltildi (READY ifadesi geri çekildi → PROVISIONAL — `BLOCKED_BY_SOURCE_QUALITY_CONTRACT`); ledger + TEST_STATUS güncellendi.

**INTEGRITY:** canonical.sqlite sha256/mtime DEĞİŞMEDİ (önce/sonra doğrulandı); microstructure.db yalnız mode=ro; migration YOK; experiment write YOK; outcome read YOK; protected delta ZERO.

## 88. LIQUIDATION SOURCE-QUALITY CONTRACT V2 — FROZEN, FIELD-LEVEL IMPLEMENTATION + CANONICAL TEST-SCOPE FREEZE (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE LIQUIDATION SOURCE-QUALITY CONTRACT V2 AND CANONICAL TEST-SCOPE FREEZE REHEARSAL" — §87'nin CONTRACT-V2 önerisi kabul edildi, METHOD_A yalnız tarihsel reconciliation kanıtı, METHOD_B ve cross-stream health REDDEDİLDİ. **Gerçek canonical migration YOK, outcome research YOK.**

**Kod:** yeni `ami/geometry/liquidation_source_quality_contract_v2.py` — `CONTRACT_VERSION="liquidation-source-quality-contract-v2"`, `ALL_MARKET_TRANSITION_TS_MS=1780767832123` (2026-06-06 17:43:52.123 UTC — ölçüldü: 40.14 günlük blackout'un [Apr 27→Jun 6] hemen sonrası ilk liq satırı, takip eden saatte 171 ayrık sembol all-market geçişini doğruluyor), `CRITICAL_GAP_MS=300_000` (collector'ın kendi frozen sabiti). **Goal D — field-level (row-level DEĞİL, operatörün açık tercihi):** 8 alanın HER BİRİ kendi required source window'unu alıyor — 6 running-cluster alanı `[bucket_start, anchor_ts]`; `running_accel` kendi frozen iki-pencere tanımıyla `[anchor_ts−2×ACCEL_WIN_SEC×1000, anchor_ts]` (bucket_start'tan BAĞIMSIZ); `inter_cluster_gap_sec` `[önceki kabul edilen anchor (veya ilk anchor ise en eski likidasyon ts), anchor_ts]` — asla mevcut bucket'ın tamlığını miras almıyor. Row-level (gerektiğinde) = 8 alanın EN KÖTÜSÜ (GAPPED>UNRESOLVED>COMPLETE), asla bağımsız bir değerlendirme değil. **Append-only field-level quality ledger:** yeni `ami_birth_truncated_geometry_field_quality_v2` tablosu + effective view (alan-bazlı en-yeni) + row-level worst-case rollup view; `(feature_id,field_name,coverage_assessment_version)` UNIQUE; farklı içerikli reassessment aynı version altında `ImmutableFieldQualityConflict` ile fail-closed; geometry değerleri/manifest'leri BAĞIMSIZ, hiç dokunulmuyor (test'le kanıtlandı).

**Testler:** yeni `tests/test_ami_geometry_liquidation_source_quality_contract_v2.py` — **15/15 ✓**, operatörün istediği 12 kanıtın hepsi: (1) pre-2026-06-06 asla COMPLETE olamaz (mükemmel sentetik cadence verilse bile), (2) gap-registry satırı YOKLUĞU tamlık ANLAMINA gelmiyor (kısa pencere geçer ama CRITICAL_GAP_MS'i aşan uzun kanıtsız pencere GAPPED), (3) ≤300s all-market cadence COMPLETE kanıtlıyor, (4) >300s cadence GAPPED üretiyor (+ayrı: resolved registry gap overlap da mükemmel cadence'e rağmen GAPPED), (5) cross-stream kanıt asla kullanılamaz (imza + gövde statik guard — docstring'deki meşru "neden reddedildi" anlatısı hariç), (6) 8 alan bağımsız pencere alıyor (running_accel ve inter_cluster_gap_sec, running_notional'dan FARKLI), (7) inter_cluster_gap_sec mevcut bucket'ın tamlığını miras ALAMIYOR (yoğun bucket + seyrek inter-anchor span senaryosu), (8) contract-v2 rerun deterministik (idempotent, content-hash birebir), (9) reassessment append-only (yeni version=INSERT, aynı version+farklı içerik=fail-closed, effective view en-yeniyi çözüyor), (10) geometry değerleri/manifest'leri quality-reassessment'lar arasında BİREBİR (2 ardışık reassessment sonrası da), (11) **gerçek veri** ile SOURCE_COMPLETE_ONLY TEST cycle N MIN_BUCKET_N=20'nin ALTINDA kalıyor (fail-closed doğrulandı), (12) hiçbir outcome kolonu okunmuyor (statik guard).

**Gerçek-veri ölçüm (`tools/research_s34_contract_v2_measurement.py`, rapor: `reports/research/s34/S34_LIQUIDATION_SOURCE_QUALITY_CONTRACT_V2_MEASUREMENT_2026-07-05.md/.json`):** 6 running-cluster alanı (running_accel dahil) = 94 COMPLETE/0 GAPPED/126 UNRESOLVED (hepsi aynı bucket-window'a bağlı). **`inter_cluster_gap_sec` sınırlayıcı alan:** 87 COMPLETE/6 GAPPED/127 UNRESOLVED — kendi (önceki-anchor→anchor) penceresi 6 resolved-gap overlap'i + 1 ek unresolved-sınır geçişi yakalıyor ki diğer 7 alan hiç görmüyor. **Row-level worst-case: SOURCE_COMPLETE=87 / SOURCE_GAPPED=6 / SOURCE_COVERAGE_UNRESOLVED=127** (operatörün yaklaşık 93/54/37-17 beklentisinden ÖLÇÜLEREK farklı — inter_cluster_gap_sec'in kendi ayrı penceresi row-level'ı 94'ten 87'ye düşürüyor; "measure, don't force" talimatına uygun, zorlanmadı). **SOURCE_COMPLETE_ONLY popülasyonu:** 87 sinyal/87 event/**51 cycle**/TRAIN=35,**TEST=16** — tamamı 2026-06 (%100 LONG_SILENCE). **MIN_BUCKET_N=20 verdict: `INSUFFICIENT_SAMPLE`** (TEST=16<20) — nitel sonuç METHOD_A/V2-yaklaşık ile AYNI kalıyor: contract-v2 altında da geçemiyor.

**Test-scope contract (Goal: canonical AMI regression):** frozen komut = `pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py` (64 dosya = 61 önceki + 3 yeni: 2 geometry + 1 contract-v2). **Collect-only #1: 798 test** (751+32+15). **Tam sequential run (32 çift, ≤2-dosya/çağrı kuralıyla): 798/798 ✓, 0 hata, 0 fail** (en yavaş çift: geometry rehearsal ikilisi, 297s gerçek-veri backfill nedeniyle). **Collect-only #2 (rerun): yine 798** — birebir. Operatörün 5 koşulu (frozen komut/tam yeşil/0 hata/ikinci collect-only=aynı sayı/TEST_STATUS'a kaydet) hepsi karşılandı.

**GERÇEK GROUND TRUTH TERFİ EDİLDİ: 751/751 → 798/798** (761 değil — `tests/test_ami_*.py` glob'u yeni 3. dosyayı da otomatik kapsıyor). `tests/legacy_tools/`'daki 3 hata kapsam DIŞI kalmaya devam ediyor (önceden-var, yapısal, AMI'yle ilgisiz — ayrıca dokümante edildi, yeni AMI failure'ı olarak SUNULMADI).

**Integrity:** canonical.sqlite sha256/mtime tüm batch boyunca BİREBİR (`c2b0b300…3098f`); microstructure.db yalnız mode=ro; migration YOK; experiment write YOK; outcome read YOK; protected delta ZERO.

**Final verdicts:**
- Source-quality contract: **`LIQUIDATION_SOURCE_QUALITY_CONTRACT_V2_FROZEN`**
- Test contract: **`AMI_REGRESSION_798_READY_AND_GREEN`** (798/798, ground truth terfi edildi)
- Research readiness: **`GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`** (SOURCE_COMPLETE_ONLY TEST cycle N=16 < MIN_BUCKET_N=20)

## 89. AMI BIRTH-TRUNCATED CASCADE GEOMETRY — CONTROLLED REAL CANONICAL MIGRATION + IMMUTABLE BACKFILL, APPLIED (2026-07-05, Sonnet 5)

Operatör onayı: "APPROVE CONTROLLED REAL CANONICAL MIGRATION — AMI BIRTH-TRUNCATED CASCADE GEOMETRY CANONICAL MIGRATION + IMMUTABLE BACKFILL". Gerçek `data/ami/canonical.sqlite`'A UYGULANDI. Detay: `MIGRATION_LOG.md` M-0030.

**Pre-migration blocking checks:**
1. Repo audit tamamlandı — migration/backfill/geometry/field-quality/provenance/effective-view'a dokunan tüm dosyalar tespit edildi.
2. **`BLOCKED_BY_QUALITY_STATE_DUPLICATION` riski bulundu ve ÇÖZÜLDÜ (kod değişikliği gerektirdi):** `ami_birth_truncated_cascade_geometry`'nin Goals A-D döneminden kalma `data_quality_status`/`coverage_assessment_version` kolonları + row-level `ami_birth_truncated_geometry_quality_assessment` tablosu/effective-view'ı, yeni field-level `ami_birth_truncated_geometry_field_quality_v2` ledger'ıyla (§88, kabul edilen authoritative contract) çakışan İKİNCİ bir quality-state mekanizmasıydı. **Seçenek 1 (remove) uygulandı** — hiçbiri gerçek DB'ye hiç migrate edilmedi:
   - `ami/geometry/birth_truncated_cascade_geometry.py` yeniden yazıldı: eski kolonlar/tablo/view kaldırıldı; `backfill()` artık `quality_status_fn` parametresi almıyor, yalnız immutable feature-value + field-provenance yazıyor.
   - `ami/geometry/birth_truncated_geometry_rehearsal.py` yeniden yazıldı: METHOD_B'nin (reddedilen) kendi gap-registry-cutoff mantığı (`classify_window_quality`/`gap_registry_cutoff_ts_ms`/`fetch_liquidation_gaps`) TAMAMEN kaldırıldı; artık `liquidation_source_quality_contract_v2`'yi (schema+classify+backfill) tek kaynak olarak çağırıyor.
   - `ami/geometry/liquidation_source_quality_contract_v2.py`'ye 2 yeni paylaşılan yardımcı eklendi: `fetch_quality_evidence()` (gap/cadence kanıtı toplama) + `assess_geometry_rows()` (zaten-backfill-edilmiş geometry satırlarını field-quality için değerlendirme) — rehearsal VE gerçek migration script'i AYNI kodu kullanıyor.
   - Tüm ilgili test dosyaları güncellendi (**21→19 cascade-geometry testi** [3 eski quality-column testi kaldırıldı, +1 yeni duplication-guard testi] — **[2026-07-05 düzeltme] önceki narrative burada yanlışlıkla "22→19" yazmıştı; gerçek pre-batch sayısı 21'di (operatörün read-only reconciliation talebiyle bulundu ve düzeltildi, aşağıdaki §90'a bakınız)**; rehearsal test dosyası güncellendi; contract-v2 test dosyasının fixture'ı stripped şemaya uyarlandı).
3. Yeni `ami/geometry/birth_truncated_geometry_canonical_migration.py:run_canonical_migration()` — `path_canonical_migration.py`'nin aynı 2-adımlı disiplini (schema önce, backfill sonra, tek auditable entry point); +3 yeni test.

**Şema (folding, disposable-doğrulanmış DDL'yle byte-for-byte eşitlik programatik kanıtlandı):** `ami/warehouse/schema.py`: `CANONICAL_SCHEMA_VERSION` **10→11**, yeni `_SCHEMA_PHASE_GEOMETRY` bloğu (`ami_birth_truncated_cascade_geometry` + `ami_birth_truncated_geometry_field_provenance` + `ami_birth_truncated_geometry_field_quality_v2` + 2 view [`..._field_quality_v2_effective`, `..._row_quality_v2_effective`]) `init_schema()`'ya kayıtlı. `tests/test_ami_lifecycle_provenance_rehearsal.py`'nin `schema_version_before in (8,9,10)` kontrolü `(8,9,10,11)`'e genişletildi (v10→v11 lifecycle semantiğini değiştirmedi, yalnız ekledi — emsal: `test_schema_fingerprint_changes_only_by_addition`).

**Disposable rehearsal ile donmuş beklenen değerler** (gerçek migration'dan HEMEN önce, `run_canonical_migration()`'ın AYNI kod yoluyla): geometry=220 satır/1760 field-quality/1760 provenance; tüm content-hash'ler kaydedildi. **Preflight (gerçek DB):** schema_version=10/events=252/signals=324/LONG=220/SHORT=104/experiment_registry=22 — operatörün "current accepted checkpoint"ıyla BİREBİR. Backup: `canonical_pre_birth_truncated_geometry_canonical_migration_20260705_141345.sqlite` (sha256=`c2b0b300…`, bağımsız mode=ro açılıp integrity_check=ok doğrulandı).

**Migration+backfill (gerçek DB'ye UYGULANDI):** schema_version=11; 220/220 sinyal (0 red); 1760 field-quality satırı (220×8, 0 conflict). **Tüm content hash'ler [geometry/manifest-set/field-quality/provenance] donmuş rehearsal değerleriyle BİREBİR eşleşti** (ayrı bağımsız doğrulama script'iyle 2. kez teyit edildi). **Field-level quality: COMPLETE=745/GAPPED=6/UNRESOLVED=1009** (operatörün "expected aggregate"ıyla birebir). **Row-level worst-case: COMPLETE=87/GAPPED=6/UNRESOLVED=127** (operatörün "current measured quality"sıyla birebir). **SOURCE_COMPLETE_ONLY: 87 sinyal/51 cycle/TRAIN=35/TEST=16 — `INSUFFICIENT_SAMPLE`.**

**Postflight — 15 maddelik bağımsız doğrulama (ayrı script, migration script'inin kendi inline assertion'larından BAĞIMSIZ 2. tur):** schema_version=11 ✓; 220 satır ✓; 0 duplicate-pair ✓; 0 future-timestamp-violation ✓; 0 feature_available_ts-mismatch ✓; tüm hash'ler eşleşti ✓; field-provenance 1760/1760 tam (8 alanın hepsi) ✓; **28 önceki canonical tablo tek tek karşılaştırıldı — 27/28 BİREBİR SABİT** (ami_events/ami_signal_lifecycle/ami_lifecycle_transitions/ami_lifecycle_path_observations/ami_lifecycle_field_provenance/ami_cycles/event_cycle_membership/ami_candles/ami_candle_morphology/ami_swings/ami_levels/ami_pushes/ami_candidate_universe/experiment_registry/experiment_results/artifact_registry/artifact_lineage/question_families/question_registry/contradiction_registry/operator_decision_queue/namespace_registry/evidence_contamination/mt_family_registry/causal_assumption_registry/data_quality_events/market_structure_versions — hepsi BİREBİR); **`researcher_exposure_ledger` +1 arttı** (1163→1164, sonra idempotent rerun'la +1 daha →1165) — `ami/research/feature_gateway.py`'nin ÖNCEDEN-VAR, by-design append-only exposure-audit mekanizması (`fetch_lifecycle_signals()`'in her çağrısında beklenen bir satır; `PROTECTED_COMPONENTS_MANIFEST.md`'de listelenmiyor, protected component DEĞİL, davranış zaten mevcut kodun kendi docstring'inde belgeli: "Every successful fetch records a researcher_exposure_ledger row"); experiment_registry/experiment_results content-hash'i (yalnız count değil, TAM içerik) BİREBİR aynı kanıtlandı ✓; foreign_key_check=0 violation ✓; integrity_check=ok ✓.

**İdempotent rerun (gerçek DB'ye 2. kez `run_canonical_migration()` çağrısı):** 0 yeni geometry/field-quality satırı, tüm content hash'ler BİREBİR (yalnız exposure-ledger +1, aynı beklenen sebep) — `NOOP_IDENTICAL` doğrulandı.

**Full regresyon — bir stale test bulundu+düzeltildi:** ilk tam koşuda `test_ami_geometry_birth_truncated_geometry_rehearsal.py::test_full_rehearsal_flow_real_data`'nın `rollback_restores_pre_migration_fingerprint` assertion'ı FAIL etti — gerçek migration artık kalıcı olduğu için bu rehearsal'ın kendi disposable-kopya rollback'i, kaynağın ZATEN sahip olduğu geometry tablolarını düşürüyor, bu yüzden fingerprint_before (artık geometry içeren gerçek kaynaktan) ile rollback-sonrası fingerprint (geometry'siz) doğal olarak eşleşmiyor — **regresyon DEĞİL**, `ami.lifecycle.migration_rehearsal`'ın `test_schema_fingerprint_changes_only_by_addition`'ıyla AYNI kategori. Modüle yeni `source_already_has_geometry_tables` flag'i eklendi, test aynı disiplinle branch-aware hale getirildi. Düzeltme sonrası **tam sequential regresyon 794/794 ✓, 0 hata** (64 dosya, ≤2-dosya/çağrı, iki bağımsız tam koşu [migration-öncesi 794/794 + migration-sonrası düzeltme-sonrası 794/794] + collect-only×3 birebir 794). **GROUND TRUTH: 798 (önceki batch) → 794** (net −4: eski quality-column testlerinin kaldırılması [−5] + yeni canonical_migration test dosyası [+3] + rehearsal dosyasındaki METHOD_B testlerinin kaldırılması dengesi).

**Backup-restore kanıtı (disposable kopyada, canlı DB'ye ASLA restore uygulanmadan):** pre-migration backup hash'i kayıtlı orijinalle (`c2b0b300…`) birebir; disposable bir kopyaya restore edilip schema_version=10 + tüm pre-migration sayıları (events=252/signals=324/LONG=220/SHORT=104/experiment_registry=22/path_observations=1466) + geometry tablosunun bu restore'da YOK olduğu (doğru pre-migration state) bağımsız doğrulandı; integrity_check=ok. Post-migration backup da alındı (`canonical_post_birth_truncated_geometry_canonical_migration_v11_20260705_144500.sqlite`, sha256=`b28b4093…`, canlı DB ile birebir).

**Integrity:** canonical.sqlite'ın gerçek hash'i migration ÖNCESİ `c2b0b300…` idi, migration+idempotent-rerun SONRASI `b28b4093…`'e değişti (beklenen — yeni tablolar + 2 exposure-ledger satırı yazıldı); her adımda sha256/mtime bağımsız doğrulandı, hiçbir adımda beklenmedik/açıklanamayan bir fark bulunmadı. microstructure.db yalnız mode=ro. Protected delta = ZERO (exposure-ledger append-only audit istisnası dışında — bu, protected component manifest'inde yok ve davranışı zaten belgeli).

**Final verdicts:**
- Migration: **`BIRTH_TRUNCATED_GEOMETRY_CANONICALIZED`**
- Test contract: **`AMI_REGRESSION_794_READY_AND_GREEN`** (yeni resmi ground truth — 798 değil, 794; sebep yukarıda açıklandı)
- Research readiness: **`GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`** (değişmedi — SOURCE_COMPLETE_ONLY TEST cycle N=16 < MIN_BUCKET_N=20; outcome deneyi HİÇ ÇALIŞTIRILMADI)

## 90. REGRESSION-GROUND-TRUTH RECONCILIATION CLOSURE — 794 → 795 (2026-07-05, Sonnet 5)

Operatör talimatı: §89'un 798→794 terfiinin read-only reconciliation'ı istendi (dosya değiştirilmeden), reconciliation'da 2 bulgu çıktı: (1) SYSTEM_STATE'in kendi §89 metni pre-batch cascade-geometry dosya sayısını yanlışlıkla "22" yazmıştı — gerçek sayı **21**'di (transcript'teki Read tool çıktısından doğrudan sayıldı, düzeltildi — yukarıdaki §89 metni + `TEST_STATUS_LATEST.md` düzeltildi); (2) `ami_birth_truncated_geometry_field_quality_v2.data_quality_status` CHECK constraint'i için raw-SQL doğrudan test kapsamı eksikti (eski, silinen row-level tablonun aynı-isimli testi vardı ama YENİ field-level tabloda hiç yoktu). Operatör bu tek eksiği KAPATMAK için minimal, kontrollü bir patch onayladı — **gerçek canonical DB'ye HİÇBİR YAZMA YOK**, yalnız test+dokümantasyon.

**Patch:** `tests/test_ami_geometry_liquidation_source_quality_contract_v2.py`'ye **+1 yeni test**: `test_schema_check_constraint_rejects_bad_data_quality_status` — hem `UPDATE ... SET data_quality_status='BOGUS_STATUS'` hem `INSERT ... VALUES (...,'BOGUS_STATUS',...)` ile `sqlite3.IntegrityError` kanıtlıyor (tek test fonksiyonu, iki assertion — node ID sayısı +1). METHOD_B testleri GERİ GETİRİLMEDİ, eski row-level quality mekanizması YENİDEN OLUŞTURULMADI, sayıyı zorlamak için fazladan test EKLENMEDİ (operatörün 3 yasağına da uyuldu).

**Tam reconciliation (dokuz kaldırılan + beş önceden-var + bir yeni):**
- **9 obsolete test kaldırıldı** (önceki batch'te): 3'ü eski row-level quality-column/tablo testleriydi (`test_backfill_rejects_unknown_quality_status`, `test_schema_check_constraint_rejects_bad_data_quality_status` [ESKİ, row-level tablo için], `test_quality_assessment_is_append_only_and_effective_view_resolves_latest`); 6'sı reddedilen METHOD_B'nin kendi fonksiyonlarının testleriydi (`test_classify_window_quality_*` ×5, `test_gap_registry_cutoff_ts_ms_is_max_start_ts`).
- **5 replacement/yeni test bu patch'ten ÖNCE zaten vardı** (önceki batch'te eklenmişti): `test_no_quality_status_column_or_table_in_this_module`, `test_module_never_reimplements_rejected_method_b_gap_cutoff_logic`, ve yeni `test_ami_geometry_birth_truncated_geometry_canonical_migration.py` dosyasının 3 testi.
- **1 SQL constraint testi BU PATCH'TE eklendi**: `test_schema_check_constraint_rejects_bad_data_quality_status` [YENİ, field-level tablo için] — kapsam boşluğunu kapatıyor.
- **Hiçbir korunan invariant artık bilerek kapsamsız değil.**

**Doğrulama:** collect-only = **795** (794+1, tek komutla ölçüldü). **Tam sequential regresyon 2× bağımsız çalıştırıldı (≤2-dosya/çağrı): Run 1 = 795/795 ✓ 0 hata; Run 2 = 795/795 ✓ 0 hata.** Frozen komut DEĞİŞMEDİ: `pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py`. canonical.sqlite sha256/mtime bu patch boyunca BİREBİR SABİT (`b28b4093…`, patch-öncesi=patch-sonrası) — **gerçek DB'ye 0 yazma**.

**798 baseline'ın kasıtlı olarak supersede edildiği açıkça belirtilir:** 798, quality-state-duplication çözümünden ÖNCEKİ (ve reddedilen METHOD_B'yi hâlâ test eden) bir ground truth'tu; 794 ve şimdi 795, o duplication'ı ÇÖZEN ve METHOD_B'yi kod tabanından TAMAMEN kaldıran refactor'ün SONRASIDIR — geriye dönük bir regresyon değil, kasıtlı bir metodoloji ilerlemesidir.

**Final verdicts:**
- Ground truth: **795/795** (794'ü supersede eder — 798'i DEĞİL, 798 zaten önceki batch'te bilinçli olarak supersede edilmişti)
- Test contract: **`AMI_REGRESSION_795_READY_AND_GREEN`**
- Research readiness: **`GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`** (değişmedi)
- Migration: değişmedi (`BIRTH_TRUNCATED_GEOMETRY_CANONICALIZED`, bu patch canonical DB'ye dokunmadı)

## 91. CVD / TAKER-VOLUME REPAIR CONTRACT + CANONICAL ANCHOR/EPOCH TANIMI (2026-07-05, Fable 5)

Operatör onayı: readiness audit'i (`S34_CVD_TAKER_VOLUME_DATA_READINESS_AUDIT_2026-07-05.md`, verdict `CVD_DATA_REPAIR_REQUIRED`) kabul edildi; sıradaki batch = **design/reconciliation/preregistration-readiness ONLY**. Gerçek DB'ye 0 yazma, 0 repair, 0 outcome; canonical.sqlite sha256/mtime batch boyunca BİREBİR SABİT doğrulandı.

**Çıktılar:** `reports/research/s34/S34_CVD_REPAIR_CONTRACT_AND_ANCHOR_DEFINITION_2026-07-05.md` + `.json` (makine-okur ayna, JSON-valid doğrulandı).

**Bu batch'te kurulan yeni read-only gerçekler:** (a) collector `_parse_agg_trade()` Binance aggregate-trade id'lerini (`a`/`f`/`l`) DÜŞÜRÜYOR — yalnız `T,s,p,q,m` saklanıyor, `ts_ms`=trade time `T` (event-time); (b) güncel collector'da aggTrades **REST fallback** var (`ingest_rest_agg_trade`, default-açık, 5s poll) ve dedup cursor'u (`_rest_agg_last_id`) yalnız in-memory → unique id kolonu olmadığından **yapısal WS/REST çift-insert riski**; (c) donmuş candle-repair `gap_manifest_pre_repair.json` (208 run / 25,716 dakika) = ETHUSDT agg_trades için otoriter **dakika-granüler coverage haritası**.

**B7 ÇÖZÜLDÜ — donmuş anchor/epoch modeli:** GLOBAL EPOCH YOK; kanonik CVD = sinyal-bazlı, event-relative, SINIRLI trailing-window net taker flow `[T−W, T]`, `T=signal_birth_ts`. Donmuş pencere ailesi: **{60s, 300s, 600s, 1800s, 3600s} + BUCKET** (geometry'nin kendi frozen bucket penceresi) = 6 pencere — gerekçe outcome'a DEĞİL mevcut 3 consumer konvansiyonuna (orderflow_lead~60s, mfe50=600s, orderflow_chart=3600s) + geometry bucket'a uzlaştırma. Reset kuralı gerekmez (stateless); restart/rejim olayları feature'da değil KALİTE statüsünde taşınır. Reddedilen alternatifler kayıtlı (fixed-clock cumulative, previous-anchor cumulative [v2'ye deferred], persisted rolling series). Symbol scope v1=yalnız ETHUSDT flow; venue=yalnız USDT-M futures; `feature_available_ts=signal_birth_ts`; as-of kuralı `ts_ms≤T` strict.

**Nicelikler (katman ayrımı yasası):** EXACT trade-level: `cvd_notional_W`/`cvd_qty_W`/`total_notional_W` (işaret: `is_buyer_maker=0`→+, `=1`→−; 3 bağımsız kod sitesinde doğrulanmış konvansiyon). PROXY: `candle_cvd_qty_W` (dakika-quantize; ASLA exact'a eşdeğer değil; kline repair yalnız dakika-net kurtarır — intra-minute sequence kline'dan kurtarılamaz). Karışık exact/proxy popülasyon YASAK (fail-closed pooling guard).

**Repairability taksonomi + atamalar:** EXACT_RECONSTRUCTABLE / PROXY_ONLY / SOURCE_GAPPED / SOURCE_COVERAGE_UNRESOLVED / UNREPAIRABLE. 6 büyük outage + registry-dönemi 20 küçük gap + ETH dakika-haritası kalıntıları → hepsi **EXACT_RECONSTRUCTABLE (PROVISIONAL** — Binance `/fapi/v1/aggTrades` + data.binance.vision; availability probe REHEARSAL'ın ilk stop-condition'ı, probe başarısız → otomatik PROXY_ONLY reclass). Sub-minute tamlık + post-07-03/B6 → SOURCE_COVERAGE_UNRESOLVED. UNREPAIRABLE: hiçbirine atanmadı (rezerve).

**Rejim segmentasyonu:** R0 (Şub15→Nis12, registry-öncesi) / R1 (Nis12→Nis24, canlı registry; SOL Nis18'de katıldı) / R2 (Nis24→Haz6 17:43, degraded — 6 büyük outage burada) / R3 (Haz6→şimdi, güncel collector + REST fallback; duplicate riski ölçülmemiş). Duplicate/ordering kuralları donduruldu: repaired satır kimliği=(symbol, agg_trade_id — repair kaynağı `a`'yı SAĞLIYOR, saklanacak); live satırlar asla düzeltilmez, yalnız supersede (path-v2-candle-repair-r1 emsali); replay=`ORDER BY ts_ms, id` deterministik.

**Versiyon ID'leri (henüz canonical SQL'de YARATILMADI):** `aggtrades-taker-side-v1` / `aggtrades-binance-fapi-repair-r1` / `s34-cvd-windowed-taker-flow-v1-birth-truncated` / `cvd-source-quality-contract-v1`.

**Eligibility (donmuş):** inferential=yalnız exact-katman + doğrulanmış-COMPLETE pencereler; PROXY_ONLY=her zaman yalnız descriptive; cycle-grouped split makinesi verbatim reuse; her outcome-öncesi MIN_BUCKET_N=20 cycle precheck.

**Önerilen implementation batch:** `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1` — D1 disposable rehearsal (tam-aralık cadence scan [B6 kapanışı], duplicate baseline, availability probe, disposable repair build, 324×6=1,944 beklenen feature satırı) / D2 validation / D3 canonical migration önerisi (schema 11→12, AYRI onay) / D4-D5 immutable backfill + append-only quality ledger. 5 stop-condition tanımlı; frozen regression 795/795 her faz sınırında yeşil kalmalı. HİÇBİRİ bu batch'te implement edilmedi.

**TEST_STATUS: NO_UPDATE_REQUIRED** (test eklenmedi/çalıştırılmadı; ground truth 795/795 aynen).

**Final verdict: `CVD_REPAIR_CONTRACT_READY_FOR_REHEARSAL`**

WAIT_FOR_OPERATOR_APPROVAL

## 92. CVD REPAIR REHEARSAL + QUALITY CONTRACT V1 — DISPOSABLE REHEARSAL (2026-07-05→06, Fable 5)

Operatör talimatı: `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1` — kabul edilmiş kontratın (§91) rehearsal fazı. Gerçek DB'ye **0 yazma**, 0 repair-backfill, 0 outcome; `data/ami/canonical.sqlite` sha256/mtime/schema_version(=11) batch boyunca BİREBİR SABİT (`b28b4093…`). Tüm yazımlar `data/ami/cvd_rehearsal_disposable_20260705/` altında disposable.

**Çıktılar:** `ami/cvd/` (4 yeni modül: `windowed_taker_flow.py`, `cvd_source_quality_contract_v1.py`, `aggtrades_repair_rehearsal.py`, `cvd_rehearsal.py`) + 3 yeni test dosyası (+52 test) + 6 rapor dosyası (`S34_CVD_REPAIR_REHEARSAL_AND_QUALITY_CONTRACT_V1_2026-07-05.md/.json`, `S34_CVD_HISTORICAL_AGGTRADES_PROBE_MANIFEST_2026-07-05.json`, `S34_CVD_CROSS_SOURCE_DEDUP_COLLISION_REPORT_2026-07-05.md`, `S34_CVD_DETERMINISTIC_REPLAY_HASH_MANIFEST_2026-07-05.json`, `S34_CVD_SCHEMA_11_TO_12_MIGRATION_PROPOSAL_2026-07-05.md`).

**Full-range scan (B6 kapandı):** Pass A (id-order) 388,452,291 satır (ETH=174.6M/BTC=193.7M/SOL=20.2M), autoincrement id'de **0 hole**. Pass B (ts-order, rejim-bazlı): dup-cluster oranları R0=0.161%/R1=0.012%/R2=0.019%/R3=0.040% — R3 elevation eşiği (10×baseline) AŞILMADI → `BLOCKED_BY_DUPLICATE_INTEGRITY` tetiklenmedi. 2026-06-06 sonrası hiçbir yeni çok-günlük outage bulunmadı (en büyük R3 gap=815.7s, zaten dondurulmuş dakika-haritasında sıfır-satır dakikası olarak açıklanıyor).

**Availability probe (Task 2):** 104 probe (6 outage×boundary/interior/end + 20 registry-gap + 4 Feb/Mar minute-map run) — **104/104 `AVAILABLE_FROM_REQUESTED_START`, 0 failed, 0 empty, 0 internal-id-hole.** Binance `/fapi/v1/aggTrades` her istenen tarihsel pencereyi (Haziran 1-5 blackout dahil) verdi.

**Cross-source dedup + replay determinism (Task 4-5):** identity `(symbol, agg_trade_id)` her extraction'da temiz (0 dup, 0 id-hole). 3 donmuş örnek pencere (blackout/healthy-R3/healthy-R0) × 2 bağımsız fetch → **hepsi content/gap/dup-manifest hash'inde birebir eşleşti** (`hard_stop_rerun_mismatch=false`). REST-vs-legacy reconciliation: healthy-R3=3050/3050 tam 1:1; healthy-R0=12257/12259 1:1 + **1 many-to-many collision** (gerçek eşzamanlı iki trade, fail-closed işaretlendi, arbitrary eşleştirme YOK — ve bu pencere zaten repair kaynağı olarak kullanılmadı çünkü sıfır eksik dakikası var).

**Ana rehearsal (Task 6-11):** 324 sinyalin 6 pencere ailesinin dokunduğu eksik dakika sayısı yalnız **35** (8 contiguous span) — kanonik popülasyon büyük outage'ların çoğuna hiç değmiyor. 8 span'ın hepsi REST'ten 2 kez bağımsız fetch edildi: **hepsi `EXACT_RECONSTRUCTED`, hepsi rerun-identical**, toplam 40,934 satır disposable staging'e yazıldı (0 immutable-conflict). Cadence threshold (source-derived): **93,195ms** (healthy-R3 residual max gap). Feature matrix: **exact=1840 + proxy=1840 + bucket_exclusion=104** → 1840+104=1944=324×6 (accounting identity tam; SESSİZ SATIR KAYBI YOK). Quality: EXACT_RECONSTRUCTABLE=1828 / SOURCE_GAPPED=12 (12'si tamamı BUCKET-pencere, cadence-fail + proxy-yok, fail-closed — bug değil). **`timestamp_violation_count=0`** (gerekli değer: 0, elde edilen: 0). BUCKET window proof: 220/220 geometry satırında start≤T ve end=T, max bucket süresi 280.2s (≤300s sınırı içinde). Coverage precheck (verbatim `w8_short_expanded_baseline` reuse): eligible=1828 satır/167 cycle (TRAIN=116/TEST=51), straddling=0, **precheck PASS** (MIN_BUCKET_N=20 her iki split'te de geçti).

**Testler (Task 12):** +52 yeni test (windowed_taker_flow=21, source_quality_contract_v1=16, repair_rehearsal=15) — sign convention, `[T−W,T]` inclusive sınır, post-T/pre-window red, aynı-ts deterministik sıralama, BUCKET determinizm, pooling guard, immutable conflict/NOOP, 5 branch'lik quality classifier sweep (UNREPAIRABLE asla otomatik atanmıyor kanıtlandı), REST pagination overlap/missing-id/retry/rerun determinizm, 5 cross-source reconciliation senaryosu (1:1/unmatched/one-to-many/many-to-one/conflicting), float/decimal fingerprint eşitliği. **Yeni ground truth: 795→847 (+52).** Collect-only ×2 (batch başı ve sonu) **birebir 847**. **2 bağımsız tam sequential regresyon (≤2-dosya/çağrı): Run1=847/847 ✓ 0 hata, Run2=847/847 ✓ 0 hata.** Frozen komut DEĞİŞMEDİ.

**Ortam notu (bu batch'in kapsamı DIŞINDA, ayrıca flagli):** C: sürücüsü %100 dolu bulundu (153MB/477GB boş) — nedeni bu oturumdan ÖNCEKİ iki eski Claude scratchpad session dizini (77GB+65GB, `D--eclipse-scalper` altında). Kullanıcıya soruldu, kendisi önce incelemek istedi — SİLİNMEDİ. Regresyon koşuları bunun yerine `--basetemp`'i D:'ye yönlendirerek (repo'ya/scratchpad'e dokunmadan) tamamlandı. Bu, repo tracking-debt kapsamının dışında, temizlenmesi gereken bir OS-seviyesi disk alanı sorunu olarak kaydedilir.

**Şema 11→12 migration önerisi (Task 13):** `S34_CVD_SCHEMA_11_TO_12_MIGRATION_PROPOSAL_2026-07-05.md` — 4 tablo (`ami_agg_trades_repaired`, `ami_cvd_repair_batch_ledger`, `ami_cvd_windowed_flow`+`_proxy`+`_bucket_exclusions`, `ami_cvd_window_quality_v1`) tam DDL/PK/FK/CHECK/immutable-version/rollback prosedürüyle. **SADECE ÖNERİ — bu batch'te UYGULANMADI, schema_version 11'de kaldı.**

**Final integrity:** canonical.sqlite sha256/mtime/schema_version batch başı=sonu BİREBİR (`b28b4093…`, 11). 0 canonical yazma, 0 outcome okuma, 0 runtime/risk/execution değişikliği. Protected delta ZERO (`tools/s34_state_machine_live_executor.py` bu oturumda hiç açılmadı/değiştirilmedi — mtime oturum öncesine ait).

**Final verdicts:**
- `CVD_REPAIR_REHEARSAL_READY_FOR_CANONICAL_MIGRATION_PROPOSAL`
- Ground truth: **847/847**
- Test contract: (yeni) — frozen komut aynı, sayı 795→847
- Migration: **PROPOSAL ONLY**, schema_version DEĞİŞMEDİ (11)

WAIT_FOR_OPERATOR_APPROVAL

## 93. CVD SCHEMA 11→12 CANONICAL MIGRATION — APPLIED (2026-07-06, Sonnet 5)

Operatör talimatı: "devam edelim" → seçilen ad: "Gerçek schema 11→12 migrasyonunu çalıştır" (§92'nin `WAIT_FOR_OPERATOR_APPROVAL` çıktığı önerinin gerçek uygulanma onayı). Öncesinde ayrı, read-only bir muhasebe-reconciliation adımı da yapıldı (feature-matrix accounting/collision reconciliation/canonical-representation freeze soruları) — verdict `CVD_MIGRATION_ROW_ACCOUNTING_FROZEN`, bu bölüm onun sonrasındaki gerçek uygulamayı kapsıyor.

**Şema fold-in:** `ami/warehouse/schema.py`'ye `_SCHEMA_PHASE_CVD` eklendi — rehearsal'ın 3 modülünün (`windowed_taker_flow.py`, `cvd_source_quality_contract_v1.py`, `aggtrades_repair_rehearsal.py`) `_SCHEMA` bloklarının birebir kopyası + önerinin izin verdiği TEK delta: 3 FK satırı (`ami_cvd_windowed_flow`/`_proxy`→signal_id/source_event_id/independent_cycle_id, `ami_cvd_bucket_exclusions`/`ami_cvd_window_quality_v1`→signal_id) + `ami_agg_trades_repaired_stage`→canonical `ami_agg_trades_repaired` rename + 2 effective-view (`ami_agg_trades_repaired_effective`, `ami_cvd_window_quality_v1_effective`). `CANONICAL_SCHEMA_VERSION` 11→12. In-memory DB'de bağımsız uygulama kanıtı: 6/6 tablo + 2/2 view mevcut, FK zinciri (ami_signal_lifecycle/ami_events/ami_cycles) çözülüyor, idempotent rerun temiz.

**Yeni `ami/cvd/cvd_canonical_migration.py`:** `run_canonical_migration(conn, source_ro)` — donmuş disposable rehearsal DB'sinden (`data/ami/cvd_rehearsal_disposable_20260705/cvd_rehearsal_disposable.sqlite`) 6 tabloyu **birebir kopyalıyor** (0 network call, 0 yeniden hesaplama — geometry migration'ın backfill-fonksiyonu çağırma desisinden farklı olarak, burada backfill zaten rehearsal'da hesaplanmış olduğu için saf içerik-kopyası + content-compare idempotency yeterli ve daha güvenli: rehearsal'ın kendi Binance-fetch'ini yeniden tetiklemiyor). Aynı-kimlik+farklı-içerik → `FrozenSourceRowConflict` (fail-closed, hiçbir zaman sessizce üzerine yazmaz).

**Disposable testler (`tests/test_ami_cvd_canonical_migration.py`, +5 test, gerçek canonical.sqlite'ın disposable KOPYASINA karşı):** not-called-automatically guard, migration rehearsal sayılarını birebir üretiyor (40934/8/1840/1840/104/1840), idempotent rerun (hash+count birebir), conflicting-content-raise, protected-invariant-unchanged (+FK/integrity check). Gerçek DB'ye dokunmadan ÖNCE hepsi yeşil.

**Backup + restore-proof:** `data/ami/backups/canonical_pre_cvd_repair_canonical_migration_20260706_065631.sqlite` (sha256 kaynakla birebir `b28b4093…`). Ayrı disposable kopyaya restore edilip GERÇEK migration'dan ÖNCE doğrulandı: schema_version=11, 6 yeni tablo YOK, events=252/signals=324/cycles=167/geometry=220, integrity_check=ok.

**Gerçek migration (canlı `data/ami/canonical.sqlite`'a uygulandı):** preflight schema_version=11 doğrulandı → `init_schema()` (v12 DDL, additive-only) → `run_canonical_migration()` (frozen source'tan kopya). Sonuç: 40,934 repaired-trade + 8 batch-ledger + 1,840 exact + 1,840 proxy + 104 bucket-exclusion + 1,840 quality satırı, **0 conflict**. Postflight: schema_version=12; **3 content hash (exact/proxy/quality) donmuş rehearsal değerleriyle BİREBİR eşleşti** (`ca11be783e6c…`/`0a8ac304fc13…`/`6e95d51a2427…`); events/signals/cycles/geometry (252/324/167/220) DEĞİŞMEDİ; **tam tablo sayımı — 33 önceki tablonun HEPSİ birebir sabit**, yalnız 6 yeni CVD tablosu eklendi (protected delta ZERO); `foreign_key_check`=[] (temiz), `integrity_check`=ok. **İdempotent rerun** (aynı gerçek DB'ye ikinci kez): 0 yeni satır, hepsi noop_identical, hash'ler değişmedi. Post-migration backup: `canonical_post_cvd_repair_canonical_migration_v12_20260706_070000.sqlite` (sha256 canlı DB'yle birebir).

**Regresyon (kendi hata + düzeltme, şeffaf kaydedildi):** İlk regresyon koşusu bu oturumda yanlışlıkla `pytest tests/test_ami_*.py ...` TEK dev pytest çağrısı olarak yapıldı — CLAUDE.md'nin kendi guardrail'ini (≤2-dosya/çağrı) ihlal ederek. Sonuç: 14 hata. Kök-neden analizi: aynı komutu doğru ≤2-dosya/çağrı paired-subprocess prosedürüyle (rehearsal batch'in kendi `run_regression.sh`'ı emsal alınarak) yeniden koşunca **13/14 hata KAYBOLDU** — bunlar yalnız tek-proses mega-invocation'ın çapraz-test-dosyası kontaminasyonuydu (asıl W8 research modülleri kendi ≤2-dosya çiftlerinde tamamen temiz), CVD migration'la İLGİSİZ. Kalan 2 gerçek/beklenen bulgu düzeltildi: (1) kendi yeni testim (`test_run_canonical_migration_against_disposable_copy_reproduces_rehearsal_counts`) artık gerçek DB'den migrate-edilmiş veri içeren bir disposable kopyaya karşı çalıştığı için branch-aware hale getirildi (geometry-migration'ın `test_schema_fingerprint_changes_only_by_addition` emsaliyle aynı disiplin — insert-branch vs all-noop-branch); (2) `test_ami_lifecycle_provenance_rehearsal.py::test_full_provenance_rehearsal_real_data`'nın `schema_version_before in (8,9,10,11)` kontrolü `(8,9,10,11,12)`'ye genişletildi (v11→v12 yalnız ADDED tablo, lifecycle/provenance semantiğini değiştirmedi — aynı emsal zinciri). Kalan 1 gözlem (`test_ami_lifecycle_short_noisy_v1_rehearsal.py::test_disposable_db_and_microstructure_db_untouched`) izole rerun'da temiz geçti — canlı collector'ın `microstructure.db`'ye sürekli yazması nedeniyle tek-seferlik ortam-zamanlaması flake'i, CVD'yle ilgisiz, kod değişmedi.

**Yeni honest ground truth: 847→852 (+5, `test_ami_cvd_canonical_migration.py`).** İki düzeltme sonrası doğru ≤2-dosya/çağrı paired-subprocess prosedürüyle **Run2=852/852 ✓ 0 hata**; collect-only=**852** (birebir). `canonical.sqlite` sha256/mtime tüm regresyon boyunca DEĞİŞMEDİ (`458bc07c…`) — testler gerçek DB'ye ek bir şey yazmadı (immutable-conflict guard'ları zaten-var-olan içerikle eşleşti).

**Final verdicts:**
- `CVD_REPAIR_REHEARSAL_CANONICAL_MIGRATION_APPLIED`
- Ground truth: **852/852**
- Migration: **APPLIED**, schema_version 11→**12**
- Protected delta: **ZERO** (yalnız 6 yeni CVD tablosu; hiçbir mevcut tablo/dosya/runtime/risk/execution değişmedi)

## 95. S34-VENGINE-V02-SHADOW-MIRROR-RUNTIME-HARDENING-V1 — INCREMENTAL/CHECKPOINT ENGINE, APPLIED (2026-07-06, Sonnet 5)

Operatör talimatı: dar kapsamlı runtime-hardening batch'i — `tools.s34_v_engine_v02_shadow_mirror`'ın her tick'te tüm geçmişi (mark_prices unbounded, liquidations unbounded) yeniden okuyup yeniden simüle etmesinin (RAM 2-3.3GB spike, tick ~40-55s) giderilmesi. Trading/research semantics değişmedi (THRESHOLD/VDEPTH/PRIOR4H/offset/wait/cross-margin sabitleri dokunulmadı); yalnız hangi geçmiş satırların yeniden okunduğu bound edildi.

**Root cause (audit):** `build_rows()` → `load_mark_index(conn,SYMBOL)` (unbounded `SELECT...ORDER BY ts_ms`, 8.77M satır ETHUSDT mark_prices) **iki kez** çağrılıyordu (bir kez direkt, bir kez transitif olarak `collect_v01_events`→`collect_events` içinde) + `load_liquidations(...,None,None)` (61,679 satır, unbounded) her tick'te sıfırdan `reconstruct_anchors` ile yeniden işleniyordu. Ledger'ın kendisi küçük (13 satır, hepsi CLOSED/FILLED, PENDING yok).

**Tasarım:** additive checkpoint (`runtime/s34_v_engine_v02_shadow_mirror_checkpoint.json`, schema_version=1, `protocol_id`+`params_fingerprint` fail-closed guard) — liquidation bucket'ları yalnız data-time watermark (`CLOSE_GRACE_SEC=3600`) kadar geride kaldığında "closed" sayılıp bir daha işlenmiyor; bir tick'te kapatılan aralık `BOOTSTRAP_CHUNK_SEC=21600` (6h) ile sınırlı (cold-start backlog'u da bounded chunk'larla replay eder — tek dev okuma yok). Açık (henüz kapanmamış) bucket + `OPEN_WINDOW_MARGIN_SEC` marjı her tick'te ucuz şekilde yeniden taranır (idempotent, aynı anchor'ı üretir → `merge_rows` no-op). Mark_prices bounded pencere: yalnız bu tick'in adaylarının ihtiyaç duyduğu `-4h-900s` (`MARK_LOOKBACK_MARGIN_SEC`) marjına kadar (`load_mark_index_range`, additive — `load_mark_index` diğer tüm çağıranlar için değişmedi). PENDING ledger satırları ham liquidation'dan değil, ledger'ın kendi persisted anchor alanlarından (`_event_from_ledger_row`) yeniden inşa edilir. `reconstruct_anchors`'a additive `seed_last_kept` parametresi eklendi (min-gap suppression state bucket-sınırında doğru devam etsin diye; default=None → tüm diğer çağıranlar için eski davranış birebir). Legacy `build_rows` (tam-geçmiş) korunur, `--full-recompute` bayrağıyla yalnız parity-doğrulama için erişilebilir; ikisi de aynı `_row_from_event` helper'ını paylaşır (satır inşası tek kaynak, drift imkânsız).

**Sertleştirme:** duplicate-instance lock (`O_CREAT|O_EXCL`, ölü-PID self-heal), ledger/brief/checkpoint atomik yazma (`os.replace`), `sqlite3.connect(...,timeout=30)`+`PRAGMA busy_timeout=30000`, tick-loop artık try/except (hata sessizce öldürmüyor — heartbeat'e `last_error` yazıp devam ediyor), SIGINT/SIGTERM handler (Windows `Stop-Process -Force` bunu bypass eder — gerçek crash-güvenliği checkpoint'in ledger-commit SONRASI ilerlemesinden geliyor, sinyalden değil). `start_eclipse.ps1`: interval 180 (bu oturumun önceki geçici mitigation'ı artık kalıcı), priority `BelowNormal` (`Start-Process` sonrası `PriorityClass` set), startup log satırı (pid/interval/db/checkpoint/priority/start_time). `status_eclipse.ps1`: uptime/priority/last_heartbeat/last_success/last_processed_ts/checkpoint_lag/tick_duration/last_error eklendi (yeni `runtime/*_state.json`+`*_checkpoint.json` alanlarından okunuyor).

**Testler:** `tests/test_s34_v_engine_v02_shadow_mirror_runtime_hardening.py`, +20 test (istenen 20 senaryonun hepsi: empty-db, bootstrap, incremental-mark, incremental-liq, same-tick-twice, second-process-same-input, restart-resume, crash-mid-tick, sqlite-busy, late-mark, out-of-order-liq, overlap-dedup, duplicate-startup, corrupt-checkpoint, fingerprint-mismatch, ledger-uniqueness, legacy-vs-incremental-parity, no-future-leakage, rows-read-bounded-as-history-grows, no-full-history-query-in-steady-state) — hepsi PASS. Regresyon: `test_s34_knowable_anchor_continuation.py`+`test_s34_v_engine_execution_frontier.py` (5 PASS), `test_s34_v_engine_shadow_observer.py` (3 PASS), `test_s34_v_engine_cancel_replace.py` (4 PASS) — hepsi ≤2-dosya/çağrı kuralına uyularak ayrı ayrı koşuldu, hiçbiri regresyon göstermedi.

**Before/after (gerçek DB, read-only, ayrı proses ölçümleri):**

| | rows read | wall time | peak working set |
|---|---:|---:|---:|
| legacy (`--full-recompute`) | ~17.6M (mark×2 + liq) | 41.1s | 3302.7 MB |
| incremental (prod ledger'dan bootstrap, ilk catch-up tick) | 143,350 (liq 590 + mark 142,760) | 0.51s | 57.5 MB |
| incremental (steady-state, restart sonrası 2. tick) | ~120,000 | 0.43s | ~33 MB (canlı proses) |

**Parity:** aynı gerçek DB üzerinde legacy 13 satır vs incremental (prod ledger'dan bootstrap) 13 satır — **`observation_id` ve TÜM alanlar birebir eşleşti** (0 mismatch, 0 missing, 0 extra).

**Restart proof (yalnız bu proses, diğer 7 collector/shadow prosesine dokunulmadı):** eski PID 22428 durduruldu (doğrulandı, `Get-Process` boş) → yeni proses başlatıldı (interval=180, priority=BelowNormal doğrulandı) → tick_duration bug'ı bulundu (başarılı tick'lerde `last_tick_duration_ms` hiç yazılmıyordu) → düzeltildi → tekrar restart (PID 12680, final). İki ardışık tick (180s arayla) doğrulandı: `rows=13 added=0` her ikisinde de (duplicate YOK), checkpoint_lag azalıyor (106.6M ms → 85.1M ms → 42.1M ms, ~13 tick'te real-time'a yetişecek), diğer 7 PID (collector_supervisor 24472, microstructure_collector 2296, bookticker 24692, event_diary 19824, heartbeat 22108, s34_live_chart 9268, s34_shadow_paper_runner 19292, s34_state_machine_shadow_runner 18808) **DEĞİŞMEDİ**.

**Kalan riskler (bugsuz iddiası YOK):** (1) `priority_class` state alanı bir tick gecikmeli self-report edebilir (kozmetik, fonksiyonel değil — gerçek OS-level priority anında doğru); (2) Windows `Stop-Process -Force` sinyal handler'ını bypass eder, graceful-shutdown kod yolu yalnız interaktif Ctrl+C'de tetiklenir (crash-safety checkpoint sıralamasından geliyor, sinyalden değil); (3) `CLOSE_GRACE_SEC=3600`/`BOOTSTRAP_CHUNK_SEC=21600` sabitleri gerçek collector reconnect/backfill gecikme dağılımına karşı sentetik fixture'la doğrulandı, prod'da haftalar boyunca gözlemlenmedi; (4) lock dosyası TOCTOU'suz (`O_CREAT|O_EXCL`) ama stale-PID kurtarma en fazla 2 deneme sınırlı.

**Rollback:** `git diff HEAD~1 -- tools/s34_v_engine_v02_shadow_mirror.py tools/research_s34_knowable_anchor_continuation.py start_eclipse.ps1 status_eclipse.ps1` ile geri alınabilir; `--full-recompute` bayrağı legacy yolu korur (rollback'e gerek kalmadan da eski davranışa tek-seferlik dönüş mümkün). `runtime/s34_v_engine_v02_shadow_mirror_checkpoint.json` silinirse bir sonraki tick mevcut ledger'dan (13 satır, immutable) güvenle yeniden bootstrap eder — evidence kaybı yok.

Verdict: **S34_VENGINE_V02_SHADOW_MIRROR_RUNTIME_HARDENING_V1_COMPLETE**. Commit (bu bölümün yazıldığı an, HEAD): `09af9dc6` (uncommitted local changes üzerine).

## 94. EPISTEMIC NULLIFIER GATES V1 — MEZARLIK + TEST-KANITI KAPILARI (2026-07-06, Fable 5)

Operatör talimatı: "bundan ilhamla eclipse_scalper'ı geliştireceksin" — validator-unlinkability tasarım path'inin (aynı oturumda üretilen P0-P6) P2 mekanizmalarının ("slash-by-nullifier" + "no-double-vote") epistemik governance'a transferi. Tespit edilen boşluk: mezarlık disiplini (`AMI_RESEARCH_PROTOCOLS.md` §5) ve TEST-kanıtı tek-kullanım yasası bugüne dek YALNIZ dokümantasyon alışkanlığıydı — `is_known_failure()` gönüllü çağrılan advisory bir substring-check, `researcher_exposure_ledger` kaydeden ama BLOKLAMAYAN bir defter. Hiçbir kod yolu mezarlık ailesinin yeniden kaydını veya aynı ailenin aynı TEST setini ikinci kez tüketmesini yapısal olarak engellemiyordu.

**Yeni `ami/governance/epistemic_gates.py` (mekanizma + seed + retro-audit; ENFORCEMENT WIRING YOK — ayrı onay):**
- **Graveyard slash-set kapısı:** `assert_not_graveyard(conn, spec_text, retry_token=None)` — knowledge.sqlite'taki yeni `graveyard_slash_fingerprints` tablosuna karşı normalize substring eşleşmesi; hit + token yok → `GraveyardRetestBlocked` (mesajda aile + kayıtlı retry_condition); hit + operatör retry-token → geçer AMA audit_log'a `GRAVEYARD_RETRY_TOKEN_USED` yazılır. Fuzzy skorlama YOK — genişlik curated listede (fail-closed: yanlış-pozitif 1 token'a mal olur, yanlış-negatif mezarlık re-testine).
- **TEST-evidence nullifier:** `consume_test_evidence()` — nullifier = sha256(family_id | split_version | sorted-TEST-cycle-set-hash), yeni `epistemic_test_nullifiers` tablosu (append-only). Yasa: ilk tüketim=CONSUMED; aynı deneyin idempotent rerun'ı=NOOP_IDENTICAL (frozen deneylerin rerun hakkı korunur, experiment_ledger immutability sözleşmesiyle uyumlu); aynı aile + aynı set + YENİ deney id → `TestEvidenceReuseBlocked`, ANCAK operatör supersession-token ile geçer ve token satıra kaydedilir (corrected-rerun emsali: -002/-003/-004 zinciri); forward-genişletilmiş set = yeni nullifier = serbest (gerçek yeni kanıt token istemez).
- **Seed:** 31 curated fingerprint (21 failure_archive satırı + FAILURE_ARCHIVE.md index'ine sonradan giren aileler [failed_cascade_SHORT, early-exit, delayed-entry, reversal harvest] + S34_ORDERFLOW_LEAD'in OFI-momentum'u). Gerçek knowledge.sqlite'a uygulandı: run1 inserted=31, rerun inserted=0 (idempotent kanıtlı).
- **Retro-audit (P0 disiplini — enforce etmeden önce ölç):** 22 mevcut experiment_registry kaydı canonical mode=ro okunarak kapıdan geçirildi: **would_block=0/22** — geçmiş disiplinin elle doğru işlediğinin ölçülmüş kanıtı; kapı ileriye dönük koruma.

**Testler:** +16 (`tests/test_ami_governance_epistemic_gates.py`) — nullifier determinizm/order-invariance/duyarlılık, tüketim yasasının 5 dalı, mezarlık blok/token-pass/audit/CLEAN/case-insensitive, seed idempotency, gerçek-veri smoke (disposable knowledge kopyası; buy-side-fade VE ofi-momentum gerçekten bloklanıyor), retro-audit read-only. **GT 852→868 (+16).** Collect-only=868 birebir; **2 bağımsız tam paired ≤2-dosya/çağrı regresyon: Run1=868/868 ✓, Run2=868/868 ✓, 0 hata.** canonical.sqlite sha256 batch boyunca DEĞİŞMEDİ (`458bc07c…`); knowledge.sqlite'a yalnız 2 additive tablo + 31 fingerprint satırı (in-pattern yazım hedefi, protected değil).

**Bilinçli SINIR (bir sonraki onay kapısı):** kapılar henüz `experiment_ledger.record_experiment_registry`'ye veya research-OS prereg akışına BAĞLANMADI — bağlama her gelecek research modülünün davranışını değiştirir, ayrı operatör onayı ister. Önerilen wiring: (a) `record_experiment_registry` INSERT dalının önüne `assert_not_graveyard(frozen_spec_birleşimi)`, (b) her `freeze_and_record()`'un TEST-okuma adımının önüne `consume_test_evidence(family, split_version, test_cycles, experiment_id)`. İleri transferler (P4 verdict-rotation/expiry yasası, P6 claim-layer downgrade etiketleri) tasarım olarak not edildi, implement edilmedi.

**Final verdicts: `EPISTEMIC_NULLIFIER_GATES_V1_MECHANISM_READY` / `AMI_REGRESSION_868_READY_AND_GREEN` — enforcement wiring: WAIT_FOR_OPERATOR_APPROVAL**

## 96. S34 PRE-CASCADE DIP-RECOVERY HİPOTEZİ — TEST EDİLDİ, NO_EDGE (2026-07-06, Fable 5)

Operatör hipotezi: "liq'ten önce 2-3 kere düşüş ve çıkış olması lazım — time framelerde test et." Yeni script `tools/research_s34_pre_cascade_dip_recovery.py` (tek proses, seri grid): ETH SELL 200K anchor evreni (625 event, 140 gün), T0 öncesi TAMAMLANMIŞ dip→recovery döngüsü (zigzag, retrace≥%60; T0'a giren tamamlanmamış cascade bacağı sayılmaz), grid = TF{1m,3m,5m,15m} × 2 lookback × amp{10,20,35,60bps}; sonuç = LONG 4h net (FEE=5bps); kronolojik %60/%40 TRAIN/TEST, TRAIN'de config seçimi, TEST'te label-shuffle permütasyon (2000 iter).

**Sonuç: NO_EDGE.** TRAIN grid işaretleri komşu configlerde ters dönüyor (tf15_n32: a20=-34.5 vs a60=+48.4; tf5_n96: a20=-28.3 vs a60=+21.7) → gürültü imzası. TRAIN-best `tf15_n32_a60` TEST'te diff=+24.7 ama perm_p=0.203; core gate (not bull/EU + silence) altında perm_p=0.411. TEST bucket deseni non-monotonik (0dip avg+21.6 / 1dip -20.6 / 2dip +48.8 / 3dip -20.0) — "2-3 dip" ayrımı tutarlı değil.

Kayıtlar: rapor `reports/research/s34/S34_PRE_CASCADE_DIP_RECOVERY.md`+`.json`; mezarlık `failure_archive` id=22 (retry: ≥6 ay ek veri + tek önceden-sabitlenmiş config prereg, TEST perm_p<0.05 şart) + `FAILURE_ARCHIVE.md` indeksine satır. Ana DB mode=ro açıldı; canonical/knowledge yazımı yalnız failure_archive INSERT (in-pattern, additive).

**Verdict: `PRE_CASCADE_DIP_RECOVERY_NO_EDGE` — deploy edilebilir bulgu yok, mevcut route'lara dokunulmadı.**

## 97. GOVERNANCE AUDIT — S34_PRE_CASCADE_DIP_RECOVERY EXECUTION PATH (2026-07-06, Fable 5)

Operatör talebi: §96'daki `tools/research_s34_pre_cascade_dip_recovery.py` çalıştırmasının M-0033 (`51e78673`) enforcement sınırından geçip geçmediğinin dar kapsamlı denetimi.

**Bulgu: script `register_experiment_with_gates`/`ami.warehouse.experiment_ledger`/`ami.governance.epistemic_gates` — hiçbirini import/çağırmadı (grep: 0 eşleşme).** `experiment_registry`/`experiment_results` (canonical.sqlite) dokunulmadı — satır sayıları öncesi/sonrası **22/323**, retro-audit'teki (§94) 22 ile birebir; canonical.sqlite sha256 `458bc07c…` DEĞİŞMEDİ. `epistemic_test_nullifiers` gerçek DB'de **0 satır**, `epistemic_authorization_tokens` tablosu gerçek DB'de **hiç yok** (M-0033 şeması yalnız disposable test kopyalarına uygulanmış — gerçek dosyaya hâlâ hiç uygulanmamış, §94/95'in "not yet applied to the real file" notuyla tutarlı). `research.sqlite` dokunulmadı.

**Tek gerçek DB yazımı:** `knowledge.sqlite` → `failure_archive` id=22 (CLAUDE.md'nin standart "reddedilenler mezarlığa" protokolü, M-0032/33'ten bağımsız, 2026-02'den beri var olan tablo/desen). Bu satır `experiment_registry` kimliği DEĞİL — `experiment_id` sütunu yok, FK yok.

**Sınıflandırma:** Script, transition-proof §12'deki 10 adı geçen legacy-bypass dosyasından (candidate_universe.py, w1-w10a) **biri değil** (grep: 0 eşleşme; ayrıca o 10 dosyanın aksine `experiment_registry`'ye inline SQL ile de yazmıyor — hiç dokunmuyor). Repo'da `experiment_registry`'ye hiç girmeyen, doğrudan `reports/research/s34/*.md+json` + mezarlığa yazan ~150+ ad hoc `research_s34_*.py` script ailesinin bir örneği (`research_s34_alpha_attribution.py`, `research_s34_mega_v1.py` ile aynı desen) — **M-0033'ün §12 kapsamının hiç hedeflemediği, önceden dokümante edilmemiş, daha geniş üçüncü bir boşluk kategorisi** (10-dosya listesinden ve `research.sqlite`/`ResearchRegistry` boşluğundan ayrı).

**Fabrikasyon yapılmadı:** family_id/split_version/nullifier bu script için hiç hesaplanmadığından, şimdi geriye dönük uydurmak "fictional preregistration/backdate" olurdu — yapılmadı. Mezarlık kaydı (id=22) olduğu gibi kaldı, geçerliliği bu denetimden etkilenmez (TRAIN/TEST disiplini kod yolu üzerinden ayrıca doğrulandı: 625 event kronolojik %60/%40 split, 32 config yalnız TRAIN'de tarandı, seçilen tek config TEST'te bir kez okundu — revizyon yok).

**Bu denetim sırasında hiçbir yeni experiment/result oluşturulmadı; hiçbir route/runtime/risk/execution/CVD canonical state değişmedi** (`execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` bu oturumda dokunulmadı).

**Önerilen genişletme (uygulanmadı, operatör onayı bekliyor):** `BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1` kapsamı, 10-dosya listesine ek olarak bu üçüncü kategoriyi (ad hoc `research_s34_*.py`/`tools/research_s34_*.py` — hiç `experiment_registry` yazmayan, mezarlık/rapor-only script'ler) ayrı ve daha geniş bir envanter maddesi olarak kaydetmeli; bunlar "gate'i atlıyor" değil "gate'in kapsamına hiç girmiyor" — kapatma stratejisi farklı olmalı (muhtemelen bu sınıfın devam eden mezarlık-only disiplini zaten yeterli, deneysel iddia/ROUTE terfi olmadıkça experiment_ledger'a sokulması gerekmez).

**Verdict: `PRE_CASCADE_DIP_RECOVERY_NO_EDGE_LEGACY_BYPASS_RECORDED` — enforced boundary'den geçmedi (hiç girmedi); §96'daki NO_EDGE bilimsel sonucu değişmeden geçerli; governance disposition = yeni, daha geniş, önceden kayıtlı olmayan boşluk kategorisi tespit edildi ve kaydedildi, kapatılmadı.**

## 98. CANONICAL OPERATIONAL HEALTH — CORRECTIVE PASS (2026-07-10, Sonnet 5)

Operatör "CANONICAL OPERATIONAL HEALTH V1 CORRECTIVE REVIEW" (provisional verdict `CANONICAL_OPERATIONAL_HEALTH_CORRECTIVE_CHANGES_REQUIRED`, 4 blocker: A single-writer, B research-fitness bounding, C test isolation, D watchdog cadence) tam kapsamlı çözüldü. Tam rapor: `reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md` (16 bölüm).

**A — Tek yazar:** `execution/health_gate.py::write_paper_trader_health` ve `tools/replay_slice.py::_write_replay_health` — ikisi de `overall.json`'a read-merge-write yapıyordu (execution/ dokunuşu için operatör sign-off bu oturumda alındı). `tools/health_state.write_overall_health` tamamen silindi; `write_component_health` artık `component="overall"` reddediyor. `tools/heartbeat_watchdog.py::build_canonical_overall` yeniden tasarlandı: eskiden önceki `overall.json`'dan "sahipsiz" component'leri (`paper_trader` vb.) okuyup taşıyordu — bu, canlıda 80 gündür (2026-04-21'den) donmuş bir `paper_trader` placeholder'ının her cycle'a kopyalanmasına yol açıyormuş (canlıda doğrulandı). Artık her component kendi dedicated dosyasından (`paper_trader.json`, `replay.json`) her cycle taze okunuyor; yoksa/bozuksa atlanıyor, asla uydurulmuyor/eskiden taşınmıyor.

**B — Research-fitness bounding:** review'ın hipotezi (`detector_heartbeat` full-scan) gerçekti ama asıl sebep değildi — gerçek sebep `tools/validate_data_research_fitness.py`'nin sınırsız `COUNT(*)` (tüm zamanlar) ve tamamen sınırsız `load_symbol_window()` (limit yok) çağrılarıydı; 792GB gerçek DB'de 90s+ timeout'a (exit 124) sebep oluyordu. `RESEARCH_FITNESS_TABLE_ALLOWLIST` (mark_prices/agg_trades/liquidations) + 600s'lik bounded recent-window + 2000 satır cap eklendi; ayrıca bu yoldaki tüm `data/microstructure.db` bağlantıları `mode=ro`'ya çevrildi (CLAUDE.md guardrail ihlaliydi). Sonuç: gerçek DB'ye karşı **1.107s**'de tamamlandı (90s+ → 1.1s).

**C — Test izolasyonu:** `tests/test_health_cycle_smoke.py` gerçek `logs/health/overall.json` + gerçek `data/microstructure.db`'ye yazıyordu/okuyordu, canlı watchdog'a sessizce bağımlıydı. `--root`/`--seed-market-data` ile tam izole edildi (in-process heartbeat_watchdog cycle, gerçek dosyalara dokunmuyor — canlı sistem çalışırken bile doğrulandı). Ayrıca ayrı, gerçek bir ürün kusuru bulundu: smoke test'in `_validate_snapshot`'ı collector'ın "status" alanının kısa kesintide "degraded" olmasını bekliyordu ama bu alan 45s staleness-gated (bağlantı state'i değil) — asla sağlanamayan yanlış bir invariant'tı, yalnızca canlı dosyaya sızıntı sayesinde "geçiyormuş" gibi görünüyordu. Düzeltildi (`connected is False` kontrolü).

**D — Watchdog cadence:** canlı sistemde pasif ölçüm (10s nominal interval) → gerçek cycle ≈11.03s (eval overhead ≈1.03s, çoğunlukla `python_process_running()`'in PowerShell spawn'ı) → 15s bütçeye karşı gerçek marj ≈3.97s. `DEFAULT_INTERVAL_SEC` 10→5 (start_eclipse.ps1 dahil); yeniden ölçüm ≈6s/cycle → ≈9s marj. Staleness boundary testleri eklendi (14.9/15.0/15.1/22.0s).

**Kontrollü aktivasyon:** yalnız `tools.heartbeat_watchdog` (PID 21352→22816, `--interval-sec 5`) döngülendi — collector/bookticker/diğer 10 proses dokunulmadı, `collection_watchdog` hiç başlatılmadı, live executor hiç açılmadı. Canlıda `paper_trader` (80 gün stale, doğru şekilde stale görünüyor) ve bu oturumun kendi `test_replay_slice.py` çalıştırmasından sızan geçici bir `replay.json` gözlemlendi (temizlendi, kaynak dosya silinince component doğru şekilde kayboldu — fabrikasyon yok).

**Test sonucu:** 241 collected, 241 passed, 0 failed (20 dosya, 13 pytest çağrısı, ≤2 dosya/çağrı).

**Verdict: `CANONICAL_OPERATIONAL_HEALTH_CORRECTED_AND_VERIFIED`.** Commit yapılmadı (operatör onayı bekliyor).

## 99. CANONICAL OPERATIONAL HEALTH — BAĞIMSIZ REVIEW + FİNAL DÜZELTME TURU (2026-07-10, Fable 5)

**Bağımsız kabul review'ı** (§98'in denetimi): çekirdek mimari canlıda bağımsız doğrulandı (research-fitness 0.510s, smoke 2×3/3, cadence 240s/41 yazım: p50=5.85s/p90=6.04s/max=6.23s, marj 8.77s) ama **2 MEDIUM bulgu** kabulü blokladı: (1) `tools/research_fitness_report.py`'nin sınırsız `--out` parametresi + kendi lokal `_atomic_write_json`'ı ile `overall.json`'a ikinci yazar teorik olarak geri getirilebilirdi (aktif değil, yapısal boşluk); (2) rapor edilen 241/241/0, aktivasyon sırasındaki sonraki bir docstring satır-sarma edit'i yüzünden güncel koda karşı 240/1'e düşmüştü (kırılgan substring assertion; sahiplik ihlali YOK). Review raporu: `reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10_INDEPENDENT_REVIEW.md`.

**Final düzeltme turu (aynı gün):** (A) `PROTECTED_OPERATIONAL_OUTPUT_BASENAMES` + `ProtectedOperationalOutputError` guard'ı `_atomic_write_json`'ın İÇİNE kondu (resolved-basename, case-insensitive, yazımdan ÖNCE reddediyor — relative/traversal/symlink/Windows-case alias'ları kapsanıyor); production CLI `--out` tamamen kaldırıldı (çıktı `logs/health/research_fitness.json`'a sabit); deprecated `collection_watchdog` wrapper'ı aynı writer'ı import ettiği için guard'ı SIFIR edit'le miras aldı (test kanıtlı). (B) Kırılgan substring testi ast-tabanlı yapısal analizle değiştirildi (yorum/docstring görünmez; health_gate'in read-only `load_overall_health` default'u tek izinli `overall.json` literal'i; docstring-sarma regresyon testi + heartbeat_watchdog pozitif-sahiplik kontrolü eklendi; kaynak okuma `utf-8-sig` — heartbeat_watchdog.py BOM taşıyor). **Final test:** son kod edit'inden SONRA tam 20-dosya seti yeniden koşuldu: **250 collected / 250 passed / 0 failed** (orijinal 241 node adıyla korunuyor ve geçiyor; +9 = 6 guard + 3 yapısal test). Canlı doğrulama: tek collector (3828) / tek watchdog (22816, restart GEREKMEDİ — değişen modüller hiçbir çalışan proseste yüklü değil) / 0 collection_watchdog / ok=GREEN / age 3.87s / native GREEN / REST fallback false / live executor OFF; korumalı-çıktı denemeleri: CLI exit 2, internal API + wrapper deterministik reject, sıfır yazım, canonical severity değişmedi. Governance düzeltmeleri: §9 unrelated-dosya listesi git status'tan türetildi (eksik `tests/test_collection_watchdog.py` eklendi), §2 haritasına `logs/health/heartbeat.json` (guardian/prometheus, ayrı dosya adı, dormant, documentation-only follow-up) ve protected-output enforcement bölümü eklendi, §10'daki bayat 241 iddiası açıkça supersede edildi. LOW carried: smoke-test subprocess env sanitization (Windows'ta minimal env kurmak kırılma riski, okunan tek env var print-eşiği — bilinçli değiştirilmedi), `_validate_snapshot` direkt unit testi.

**Verdict: `CANONICAL_OPERATIONAL_HEALTH_CORRECTED_AND_VERIFIED`.** Commit yapılmadı — final bağımsız kabul bekliyor.

## 100. CANONICAL OPERATIONAL HEALTH — COMMIT CLOSURE (2026-07-10, Sonnet 5)

**Final bağımsız kabul review'ı** (§99'un çıktısını denetledi): 0 HIGH, 0 MEDIUM; orijinal 241 küme 12/9-deselect ile ayrıştırılarak **241/241** kanıtlandı; genişletilmiş küme bağımsız yeniden koşuldu **250/250/0**; research-fitness üretim tekrarı 0.500s (exit 0); AST tabanlı repo-geneli tek-yazar denetimi 1012 non-test modülde 0 ihlal buldu (symlink + NTFS 8.3 short-name alias denemeleri dahil — ikisi de reddedildi); canlı durum sağlıklı (collector 3828 / watchdog 22816, restart yok). **Verdict: `CANONICAL_OPERATIONAL_HEALTH_ACCEPTED_WITH_LOW_FINDINGS`** (2 LOW carried: smoke-test subprocess env inheritance, `_validate_snapshot` doğrudan unit testi yok — ikisi de bilinçli, gerekçeli).

**4 commit + bu governance commit'i ile kapatıldı** (`codex/data-layer-fallback-cleanup` dalı, hiçbiri henüz push edilmedi):

| # | Hash | Konu | Dosya sayısı |
|---|---|---|---|
| 1 | `00ef49ad5d0d8a94acaba76795fadddec5c98534` | `feat(health): enforce single-writer canonical health aggregation` | 10 |
| 2 | `81ec6d7139b4148b237fc54244c182e0c427cbc1` | `feat(research-fitness): bound read-only evaluation and protect health outputs` | 9 |
| 3 | `f3d95f5eb161e828cd106581e997056667d6c40d` | `test(health): isolate health-cycle smoke from live runtime` | 2 |
| 4 | `6faa2177648d40ecf147392d1848405282294933` | `chore(health): set watchdog interval from measured freshness budget` | 1 (start_eclipse.ps1, tek satır) |

Patch-seviyesi izolasyon: `tools/heartbeat_watchdog.py`, `data/microstructure_collector.py`, `tests/test_heartbeat_watchdog.py`'nin "interleaved native-WS hunk'ı olabilir" şüphesi commit öncesi hunk-hunk incelemeyle YANLIŞ çıktı — üçünün de tüm diff'i zaten single-writer kapsamındaydı (native-WS kodu bu diff'lerde sadece değişmeyen context satırı olarak görünüyor, HEAD'de zaten var); hiçbir `git add -p` gerekmedi, dosyalar bütün olarak stage edildi.

**Kapsam dışı bırakılan (dokunulmadı, hâlâ dirty):** `tools/native_ws_health_policy.py`, `tests/test_native_ws_health_policy.py`, `tools/s34_cascade_navigation_dashboard.py`, `tools/s34_realtime_shadow_runner.py`, `status_eclipse.ps1`, `stop_eclipse.ps1`, `runtime/dashboard_backend.json` (silme), `TEST_STATUS_LATEST.md` — hepsi git status'tan doğrulandı, hiçbiri 5 commit'e girmedi.

**Post-commit doğrulama:** tam 20-dosya seti son commit'ten SONRA yeniden koşuldu — **250/250/0**; research-fitness tekrar tekrar üretimde çalıştı (mode=ro, DB/checkpoint mutasyonu yok); canlı: 1 collector/1 watchdog/0 collection_watchdog/0 live executor/0 duplicate, overall=ok⇔WATCHDOG=GREEN, native GREEN, REST fallback false, kaynak tabloları taze; hiçbir proses bu kapanış tarafından restart edilmedi (watchdog restart'ı önceki turda, §98'de, zaten yapılmıştı). Staging area boş, 5 commit HEAD ancestor'ı.

**Verdict: `CANONICAL_OPERATIONAL_HEALTH_COMMITTED_AND_CLOSED`.** Push YAPILMADI (operatör talebi dışında).

## 101. OD-SWEEP — KARAR KUYRUĞU TOPLU YÜRÜTME + MIN-GAP PERSISTENT-V2 (2026-07-10, Fable 5)

Operatör onayı: "hepsini yap sonuçları getir" (karar kuyruğu + 2026-07-10 cross-engine/parite audit bulguları üzerine). Kapsam dışı tutulanlar: diğer oturumun sahiplendiği canonical-health dosyaları, `start/stop/status_eclipse.ps1`, collector; hiçbir proses restart edilmedi, hiçbir live executor açılmadı.

**1) OD-018 — Paper runner min-gap `persistent-v2` (UYGULANDI).** Kanıtlanan defekt: loop modunda `last_signal_ms` her `run_once`'ta −∞'dan başlıyordu → 900s bağımsız-döngü aralığı cycle'lar arası UNUTULUYORDU (backfill 299–899s'de bastırırken live loop hepsini kabul ediyordu; gerçek `_bucket_events` ile in-memory repro). Fix (`tools/s34_shadow_paper_runner.py`): (a) `MIN_GAP_SEMANTICS_VERSION="persistent-v2"`; (b) `_bucket_events(..., last_signal_ms_seed)` — emsal: `reconstruct_anchors.seed_last_kept` (mirror runtime-hardening); (c) state dosyasına `last_signal_ts_ms_by_rule` haritası (rule-başına SON EMİT EDİLEN sinyalin ts'i; downstream akıbetten bağımsız — regime-red/governance-arşiv/NO_FILL de gap tüketir, veri-deterministik referans); (d) bucket-hizalı rescan (`scan_start = floor_to_bucket(cursor)`) — cursor bucket ortasındayken cluster kimliği artık bölünmüyor (pre-v2'de eşik geçişi TAMAMEN KAÇIRILABİLİYORDU — testle kanıtlandı); signal_key dedup re-emisyonları emiyor. Sinyal/status/state çıktılarına `min_gap_semantics` etiketi eklendi. **Pre-v2 canlı paper popülasyonu invocation-local semantikle üretildi — 2026-07-10 sınırı üzerinden HAVUZLANMAZ (forward N=0).** Test: `tests/test_s34_shadow_paper_min_gap_parity.py` 6/6 ✓. Çalışan proses (PID 24468) eski kodla devam ediyor; v2 operatör restart'ında devreye girer.

**2) OD-004 — `bd_first_buy50` exit gözlemcisi (UYGULANDI).** `tools/s34_realtime_shadow_runner.py` BUYF yoluna observation-only gözlemci: T+30m sonrası ilk yeni ETH BUY≥50K liq'te hipotetik çıkış (mark), kapanışta `delta_vs_baseline_bps`; SİPARİŞ YOK, baseline çıkış DEĞİŞMEDİ. Aktivasyon-ts frozen: `2026-07-10T18:00:00Z`. `--once` dry-run bilerek atlandı (prod prosesi aynı state'e canlı yazıyor); restart'ta devreye girer.

**3) OD-007 — kısmi.** Kuyruk öncülü yanlıştı: `data/test_s34_*` 414 dosyanın yalnız 6'sı 0-byte idi → 6'sı silindi; kalan 408 dosya 36KB dolu DB (~15MB) — veri-silme kuralı gereği DOKUNULMADI, ayrı onay ister.

**4) OD-011 — ilk-geçiş (salt-okunur; knowledge.sqlite'a YAZILMADI).** 252 event'in 252'si canonical-v1'e maplenmiş, 167 cycle (deflasyon 0.66). ~~**HOUR17 popülasyonu: 68 event → 53 bağımsız cycle (deflasyon 0.78)**~~ **`INVALID_WRONG_POPULATION` (2026-07-11 düzeltmesi, bkz. §104):** bu rakam `ami_events`'i route'tan bağımsız yalnız "saat≥17" ile filtrelemekten üretilmişti — `ami_events`'te HİÇ HOUR17-etiketli satır yok, gerçek K-S34-HOUR17-001 kanıtıyla ilgisi yok, HOUR17 kararlarında KULLANILMASIN. Doğru rakam: **127 event → 93 bağımsız cycle (deflasyon 0.732)** — §104. ~~**MONDAY-VETO: ... recompute BLOCKED_BY_SOURCE**~~ **DÜZELTİLDİ (2026-07-11, §106):** bu da aynı yanlış-ami_events hatasıydı — MONDAY-VETO gerçek hour17 popülasyonundan reprodükte EDİLEBİLİYORDU (Pazartesi N=15 cycle WR27% −49.9bps, `DESCRIPTIVE_ONLY_LOW_SAMPLE`). Kalan 6 KO da §106'da ele alındı; tam recompute dalgası TAMAMLANDI.

**5) Karar kaydı güncellemeleri:** OD-005 ANSWERED (gözlemde tut; değerlendirme ~2027-01; not: 2026-07-10 forward n=2 BUYF +145.05/+12.57 net — anekdot, karar değiştirmez), OD-008 ANSWERED (prereg'ler AÇILMADI; ≥6 ay forward şartı, ~2027-01), OD-009 ANSWERED (backlog'da Phase 6 adayı; prereg dalga açılınca yazılır). OD-001 OPEN (operatör dışı kaynak bilgisi gerekiyor), OD-002 ANSWERED (politika: v0.3=content-canonical, PATCH zinciri v0.2'de + v0.3'e senkron not), OD-006 OPEN (collector aktivasyonu bu batch'te BLOKE: `start_eclipse.ps1` diğer oturumun kapsamındaydı + kalıcı proses operatör dışarıdan başlatmalı; collector implementasyonu ayrı batch), OD-010 PARTIAL (bu batch'in kendi dosyaları commit'lendi; önceden var olan untracked canonical doc kümesi küratörlü liste ister), OD-013..017 OPEN (ay-ölçeği altyapı dalgaları — tek batch'te yürütülemez).

**6) Cross-engine opportunity contract (kanonik özet).** Shadow (`s34_realtime_shadow_runner`) ve paper (`s34_shadow_paper_runner`) motorlarının fırsat kimlikleri KASITLI OLARAK EŞDEĞER DEĞİLDİR: anahtarlar `SHD:{anchor_ts}:{route}` vs `{rule}:{bucket}`; eşikler 100K/200K vs 50K–500K; evren ETH-only vs ETH+SOL; fill mark+5bps flat vs book-ticker executable+4bps bacak; çıkış fixed-horizon (TP/BE YOK, tasarım gereği) vs TP/SL/BE. **Sonuçları doğrudan karşılaştırılamaz.** Ortak fırsat anahtarı: `(symbol, liq_side, bucket_id[300s], threshold)`; pozisyon karşılaştırması için + `(direction, rule_or_route@version)`. `max_open_trades=1` kapsamı RULE-BAŞINA'dır; ayrı bir aynı-sembol+aynı-yön kapısı vardır; GLOBAL portföy limiti YOKTUR (2026-07-10 ETH+SOL eşzamanlı SL çifti bu tasarımın sonucudur). İsim değişikliği yapılmadı; scope metadata bir sonraki rule-set versiyonunda.

**7) Doğrulama:** `py_compile` 2/2 ✓; yeni test 6/6 ✓; guardrail'ler: ≤2 test dosyası/çağrı ✓, basetemp scratchpad ✓, ana DB'ye yalnız mode=ro ✓, paralel proses yok ✓, protected dosyalara sıfır dokunuş ✓.

## 102. OD-SWEEP ACTIVATION CORRECTIVE — MİGRASYON + BD_FIRST_BUY50 SINIR DÜZELTMESİ (2026-07-10, Sonnet 5)

Bağımsız inceleme (§101'in commit'leri üzerine, `02a2fc30` + `fdf3e84f`) **`OD_SWEEP_CORRECTIVE_CHANGES_REQUIRED`** verdiğini kanıtladı: 2 gerçek, reprodüklenmiş MEDIUM bulgu. Bu bölüm o düzeltmeleri kayda geçirir. Hiçbir proses restart edilmedi, hiçbir live executor açılmadı, hiçbir DB/checkpoint/trade-store/PID/log dosyası değiştirilmedi.

**Commit provenance düzeltmesi (doğru kayıt):** `02a2fc30`, `tools/s34_shadow_paper_runner.py`'nin daha önce hiç git'e girmemiş halinin **ilk kez tam-dosya yakalaması** oldu (OD-018 persistent-v2 bu yakalamanın içine serpiştirildi — commit mesajı bunu artımlı bir diff gibi ifade ediyordu, değildi). `fdf3e84f` de aynı şekilde `tools/s34_realtime_shadow_runner.py`'nin önceden commit'lenmemiş sürüklenmesinin ilk yakalamasıydı + OD-004. **Hiçbiri, o anda çalışan bir Python prosesinin bellekteki kaynağıyla bayt-özdeş olduğunu iddia etmez** — süreç import sonrası kaynağı yeniden yüklemez; her iki dosya da bu batch'ten önceki oturum(lar)ın commit'lenmemiş değişiklikleriyle bu oturumun eklemelerinin birleşimidir. Her iki commit korunuyor (amend/squash/rebase YAPILMADI), üstüne dar kapsamlı düzeltme commit'leri eklendi.

**Bulgu 1 (MEDIUM, düzeltildi) — persistent-v2 ilk-aktivasyon migrasyonu.** Kanıt: `last_signal_ts_ms_by_rule` ilk v2 çalışmasında boştu; bucket-hizalı rescan yalnız persisted cursor'ın bucket'ından başlıyordu → gerçek son pre-v2 emisyonun bucket'ı zaten cursor'ın gerisinde kalmışsa (normal operasyonda neredeyse her zaman), o emisyon "unutuluyor" ve restart sonrası 900s'den KISA bir aralıktaki yeni sinyal YANLIŞLIKLA KABUL edilebiliyordu (100s/899s'de reprodüklendi; 900s/901s'de "doğru ama tesadüfen doğru"ydu). **Düzeltme** (`tools/s34_shadow_paper_runner.py`): `_derive_min_gap_seed_from_history()` — zaten bellekte olan `trades` deposundan (dosya değil, `run_once`'ın kendi yüklediği yapı) rule-başına son emisyon ts'sini tek seferlik türetir. Emisyon-tam-günlük kanıtı: `run_once`'ın aday döngüsündeki HER dal (deprecated/no-fill/regime/same-cluster/quality-gate/risk-gate/accepted) `_bucket_events`'in döndürdüğü her sinyal için tam olarak bir trade kaydı (OPEN/CLOSED/SKIPPED) yazar, gating'den ÖNCE — dolayısıyla `max(signal_ts_ms)` filtrelenmiş `rule.name` başına `_bucket_events`'in iç `last_signal_ms`'ini sadakatle yeniden inşa eder. Gerçek üretim verisiyle doğrulandı (salt-okunur): 1338 trade, 12 rule adı, **0 ambiguous identity, 0 malformed, tüm max(signal_ts_ms) ≤ cursor** — migrasyon gerçek veriyle ilk çalıştığında hiçbir rule fail-closed olmayacak. Durum makinesi: `DERIVED_FROM_HISTORY` / `NO_PRIOR_EMISSION` (hata değil, taze rule) / `AMBIGUOUS_FAILED` (aynı rule.name farklı symbol/threshold/liq_side altında görülürse veya `signal_ts_ms` eksikse — o rule için YENİ SİNYAL ÜRETİLMEZ, operatör manuel çözene kadar; arşivlenmiş/donmuş rule'lar başka bir rule'u asla seed edemez, isim eşleşmesi kesin). Migrasyon rule-başına TEK SEFER çalışır (`min_gap_state_provenance_by_rule` varlığıyla korunur — restart sonrası zaten migrate olmuş bir rule asla yeniden türetilmez, persisted değer kullanılır); state dosyasına aynı atomik `_write_json` çağrısıyla yazılır (`min_gap_state_migration_version`, `min_gap_state_initialized_at_utc`, `min_gap_state_provenance_by_rule`). Test: `tests/test_s34_shadow_paper_min_gap_migration.py` **16/16 ✓** (100/899/900/901s sınır + several-buckets-behind + cursor-straddle + restart-parity + idempotent + no-prior-emission + ambiguous-fail-closed×2 + rule-izolasyonu + governance/regime/no-fill/accepted hepsi aynı şekilde gap tüketiyor + loop=backfill (v1→v2 sınırı üzerinden) + duplicate-yok + top-level `min_gap_semantics` alanı). Orijinal 6/6 parite paketi regresyonsuz yeniden çalıştırıldı.

**Bulgu 1-yan (LOW, düzeltildi) — protokol versiyonu görünürlüğü.** `trade["min_gap_semantics"]` artık **top-level** alan (önceden yalnız `trade["signal"]["min_gap_semantics"]` içinde iç içeydi); gerçek 1338 pre-v2 trade'in HİÇBİRİnde (0/1338, hiçbir derinlikte) bu alan yok — pre-v2/v2 ayrımı tek `.get()` ile filtrelenebilir, doğrulandı.

**Bulgu 2 (MEDIUM, düzeltildi) — bd_first_buy50 aktivasyon sınırı zorlanmıyordu.** Kanıt: `BD_FIRST_BUY50_ACTIVATION_UTC` yalnız metadata etiketi olarak saklanıyordu; kodda `now_ms >= activation_ts` kontrolü YOKTU — restart sonrası hâlâ açık bir `OPEN_BUY_FADE` pozisyonu olsaydı, gözlemci restart-öncesi bir BUY≥50K olayını "yeni gözlemlenmiş" gibi geriye dönük damgalayabilirdi (şu an istismar edilemez: canlı shadow state'te 0 açık BUYF pozisyonu var, ama kod defekti gerçek). **Düzeltme** (`tools/s34_realtime_shadow_runner.py`): `BD_FIRST_BUY50_ACTIVATION_MS` (ISO'dan bir kez ms'e çevrilir) artık **zorlanan olay-zamanı alt sınırı**: arama penceresi `max(entry_ts_ms + 30dk, BD_FIRST_BUY50_ACTIVATION_MS)`'den ÖNCE asla başlamaz — restart zamanlaması hiçbir zaman ekonomik sınır değildir. Alanlar yeniden adlandırıldı/eklendi: `hypothetical_exit_ts_ms` (gerçek piyasa-olay ts'si, `shadow_exit_ts_ms`'in yerine), `detected_at_ts_ms` (bu proses ne zaman hesapladı), `reconstructed_after_restart` (olay `_PROCESS_START_MS`'den önce mi — bu proses ömrü boyunca sabit, import-anı damgası), `protocol_version`, `route`, kapanışta `baseline_exit_ts_ms`/`baseline_net_bps`/`delta_vs_baseline_bps`. Pozisyon yaşam döngüsü alanlarına (status/entry/exit/TP-SL-TIME) hâlâ SIFIR dokunuş — yalnız kendi alt-nesnesini yazıyor. Test: `tests/test_s34_bd_first_buy50_observer.py` **12/12 ✓** (aktivasyon-öncesi/sonrası olay ayrımı, tam sınır ±1ms, entry+30dk sınırı, restart-idempotence, reconstructed-vs-live damgalama, no-event, non-BUYF pozisyon dokunulmazlığı, canonical TIME_EXIT değişmedi, yaşam-döngüsü alanları mutasyona kapalı). `log_event` her testte no-op'a monkeypatch'lendi — canlı `reports/shadow/s34_state_machine_shadow.jsonl` dosyasına SIFIR yazma (doğrulandı).

**Toplam yeni test: 16+12=28 (2 yeni dosya) + orijinal 6 (regresyonsuz) = 34/34 ✓.** `py_compile` iki dosya için ✓. **Canlı durum değişmedi:** PID'ler birebir aynı (collector 3828/supervisor 23052/paper 24468/watchdog 22816/realtime-shadow 4728/bookticker 5624), her iki live-executor PID dosyası hâlâ `0`, watchdog `overall=GREEN native_ws=GREEN`. Staging alanı boş; kapsam dışı diğer oturum dosyaları (`status/stop_eclipse.ps1`, `native_ws_health_policy.py`+testi, `s34_cascade_navigation_dashboard.py`, `runtime/dashboard_backend.json` silme) dokunulmadan bırakıldı, commit'lenmedi.

**Verdict: `OD_SWEEP_ACTIVATION_CORRECTED_AND_VERIFIED`.** persistent-v2 ve bd_first_buy50 hâlâ **inaktif** — yalnız operatör restart'ında devreye girer. Push YAPILMADI.

## 103. OD-SWEEP CONTROLLED ACTIVATION — HEDEFLİ RESTART + 10-DK SOAK (2026-07-10, Opus 4.8)

Operatör açık yetkisi ("execute the already-reviewed targeted restart yourself now") ile, bağımsız kabul verdicti `OD_SWEEP_ACCEPTED_WITH_LOW_FINDINGS_READY_FOR_CONTROLLED_ACTIVATION` sonrası persistent-v2 + bd_first_buy50 **AKTİF EDİLDİ**. Yalnız 2 runner değiştirildi; tam-stack restart YAPILMADI, `stop_eclipse.ps1` (dirty) KULLANILMADI, live executor açılmadı.

**Dayanıklılık önkanıtı:** restart öncesi tek-kullanımlık detached probe (Start-Process -WindowStyle Hidden, start_eclipse.ps1 ile aynı mekanizma) spawn edildi → ayrı tool-call sınırında hayatta kaldı → sandbox'ın detached prosesleri öldürmediği KANITLANDI (CLAUDE.md'nin genel uyarısına rağmen, bu oturum için ampirik). Ancak sonuç yine de operatör onayıyla yürütüldü.

**Restart mekanizması:** `start_eclipse.ps1` **temiz/HEAD ile birebir** (diğer oturumun dirty stop/status dosyalarından bağımsız). Kritik: bare `start_eclipse.ps1` (satır 212) `-NoCleanStop` verilmezse dirty `stop_eclipse.ps1`'i çağırıp tam-stack durdururdu → **`-NoCleanStop` ile çalıştırıldı** (tek stop_eclipse referansı bu bayrakla atlanır). `Start-RegisteredPythonProcess` idempotent: canlı-PID'li her rol (collector/supervisor/bookticker/watchdog/oi-poller/chart/mirror/replay/event-diary) `ALREADY_RUNNING` ile atlandı; yalnız durdurduğum 2 rol fresh `Start-Process` aldı.

**PID geçişleri:** paper `24468→19504` (STARTED), state-machine-shadow `4728→21576` (STARTED). STOP_UTC=20:21:47Z, START_UTC=20:21:49Z. Eski PID'ler gitti, yeni PID'ler ayrı tool-call'da hayatta (dayanıklılık kanıtlandı). Diğer 10 rol PID'i DEĞİŞMEDİ. Duplicate=0. Live executor pid dosyaları 0/0, süreç taraması 0.

**Paper first-tick migrasyon (ilk çevrim 20:21:50Z):** `min_gap_semantics=persistent-v2`, `min_gap_state_migration_version=v1-derived-from-trade-history`, `min_gap_state_initialized_at_utc=2026-07-10T20:21:50.127674Z`. **12/12 rule `DERIVED_FROM_HISTORY`** (0 AMBIGUOUS_FAILED, 0 NO_PRIOR_EMISSION); her rule'un `seed_ts_ms`=`last_signal_ts_ms_by_rule`=bağımsız türetilen pre-restart son emisyon ts'i (12/12 birebir eşleşme). Cursor monotonik: `1783713369660` (=19:56:09Z, sabit — DB'deki en son ETHUSDT likidasyonu; ETH ~40dk sessiz, mark akıyor ama `end_ms=min(latest_liq,latest_mark)` likidasyona sabitli; restart-öncesi de aynıydı → regresyon DEĞİL, "sessiz piyasa ≠ takılı consumer"). State her 60s yeniden yazılıyor (15 çevrim, 0 stderr).

**Population sınırı:** pre-v2 final=1339 kayıt (son id P1339), ilk olası v2 id=P1340. v2 N restart-anında=0, soak-sonrası=0 (sessiz piyasada yeni sinyal yok — doğru). Duplicate trade_id=0. Tüm 1339 pre-v2 kaydı DEĞİŞMEDİ ve etiketsiz (top-level `min_gap_semantics` yalnız yeni kayıtlarda görünecek; henüz 0).

**Realtime-shadow (bd_first_buy50):** yeni runner 21576 committed bfe56930 kodunu yükledi; aktivasyon sınırı `max(entry+30dk, 2026-07-10T18:00:00Z)` zorlanıyor. 2 açık `LONG_HOUR17` pozisyonu (H17/H17C) restart boyunca birebir korundu: entry_ts_ms=1783709658317, entry_price=1782.1881938, exit_due_ms=1783731258317, status=OPEN — hepsi baseline ile aynı. 0 açık BUYF → gözlemci kaydı beklenmez (yok, doğru). Hiçbir kapalı pozisyon değişmedi, hiçbir canonical yaşam-döngüsü alanı gözlemci başlangıcından etkilenmedi.

**10-dk soak (T+0..T+10, 6 checkpoint):** her checkpoint'te paper 19504 + shadow 21576 CANLI (x1), health=GREEN, native_ws=GREEN, rest_fallback inactive, shadow state ilerliyor (20:26:36→20:36:54), executors 0/0, duplicate yok, beklenmedik trade patlaması yok (total 1339 sabit). Post-soak: 12 rol topolojisi değişmedi, migrasyon provenance stabil (initialized_at 20:21:50'de sabit → migrasyon TEK kez çalıştı, idempotent).

**Verdict: `OD_SWEEP_CONTROLLED_ACTIVATION_COMPLETE`.** persistent-v2 + bd_first_buy50 CANLI. Bekleyen LOW bulgular ileriye taşındı (rule-body in-place revizyonu için gelecekte `rule_version`; state yazımı tek-JSON, torn-write'a karşı pre-existing). Push YAPILMADI.

## 104. HOUR17 CYCLE-ADJUSTED RECOMPUTE + MAYIS 2026 LİKİDASYON BOŞLUĞU ADLİ İNCELEMESİ (2026-07-11, Sonnet 5)

Tam rapor: `reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md`. Salt-okunur bilimsel recompute + adli inceleme; hiçbir kod/route/eşik değişmedi, hiçbir proses restart edilmedi, `knowledge.sqlite`/`canonical.sqlite` DEĞİŞTİRİLMEDİ.

**1) Eski "68→53" rakamı `INVALID_WRONG_POPULATION`** — kanıt: `ami_events`'te sıfır HOUR17-etiketli satır (route_version'lar hep pre-HOUR17 adları); rakam route'tan bağımsız kaba "saat≥17" filtresinden üretilmişti. Tarihsel kayıt korunuyor, kullanılmasın.

**2) Doğru recompute:** KO `K-S34-HOUR17-001`'in headline'ı (avg+40.8/wr61.5%) aslında **150K** eşiğin (`S5_150K_pred_FULL`) rakamları — hem KO metni hem deploy edilen `LONG_HOUR17_HOLD6H` **200K+hour≥17** kullanıyor, tam eşleşen yayınlanmış konfig **`S6_200K_full`** (N=126, WR65.1%, avg+39.8). `tools/research_s34_silence_predictor.py` metodolojisi `data/microstructure.db` (mode=ro) üzerinden birebir reprodükte edildi (N=127≈126 doğrulandı), `ami/identity/cycle_resolver.py` canonical-v1 BİREBİR import edilerek uygulandı: **127 event → 93 bağımsız cycle (deflasyon 0.732)**. Cycle-düzeyi: WR=62.4%, mean=+32.47bps, median=+24.04bps, cum=+3019.4bps. Holdout train'i geçiyor (decay yok); hafta-blok bootstrap P(mean>0)=0.986; 3× maliyet stresine dayanıklı (fee15bps'te hâlâ +22.5bps); ama top5-cycle payı %54.5 (yoğunlaşma). İleri shadow N=1 (net -57bps, anlamsız tekil gözlem). **Sınıflandırma: `HOUR17_HISTORICAL_CYCLE_EVIDENCE_POSITIVE_BUT_FRAGILE`.**

**3) Mayıs boşluğu adli bulgusu:** Gerçek boşluk **2026-04-27T14:27Z → 2026-06-06T17:47Z (~40gün3sa)**, üç sembol de (ETH/BTC/SOL) saniyeler içinde birlikte başlayıp bitiyor → tek paylaşılan neden. `logs/diagnostics/endpoint_matrix_result.json` (2026-05-03, dönem-içi first-party diagnostik): 6 endpoint konfigürasyonu (liquidation + **control** markPrice/aggTrade dahil) hepsi `verdict=DEAD` (0 frame, bağlantı kurulmasına rağmen) → araştırmacının kendi sonucu: *"network/VPN path blocker"*. Arşivde (`data/archives/raw_v1`) `liquidations` tablosu hiç yok → veri taşınmadı/silinmedi, hiç toplanmadı. **Kök-neden: `ALL_LIQUIDATION_SYMBOLS_GAP` + `TRANSPORT_SPECIFIC_OUTAGE` — kategori kanıtlı, kesin nihai neden/düzeltme KAPATILMADI.** "4.5 ay sürekli kapsama" ifadesi YANLIŞ — gerçek ≈3.5 ay + 40 günlük kör nokta; Mayıs'ta 0 cycle "kanıtlanmış sıfır cascade" değil, veri yokluğu.

**Verdict: `HOUR17_GOVERNANCE_CORRECTED_MAY_GAP_CONFIRMED_ROOT_CAUSE_UNRESOLVED`.** Terfi/route değişikliği YOK, route donmuş kalıyor. Push YAPILMADI.

## 105. K-S34-HOUR17-001 KO DÜZELTMESİ (data/ami/knowledge.sqlite) + KÖK-NEDEN DERİNLEŞTİRME (2026-07-11, Sonnet 5)

Operatör açık yetkisiyle **tek satır, tek KO** (`K-S34-HOUR17-001`) `data/ami/knowledge.sqlite`'ta düzeltildi — kanonik `ami/knowledge/store.py:KnowledgeStore.put()` + `ami/knowledge/objects.py:KnowledgeObject.touch_version()` API'si üzerinden (el yazması SQL YOK). DB gitignore'lu/untracked (`data/**/*.sqlite`), dolayısıyla bu değişikliğin git commit'i YOK — kanıtlar burada ve yeni raporda kayıtlı.

**Doğrulama:** DB önce sha256=`710b3f68…3dc65` (110592B) → sonra `24db6329…27760` (114688B, WAL büyümesi kadar). `PRAGMA integrity_check`=ok, şema/`user_version` DEĞİŞMEDİ. 11 knowledge satırının **10'u payload-hash birebir DEĞİŞMEDİ**, yalnız `K-S34-HOUR17-001` değişti (satır sayısı=1, doğrulandı). `audit_log`'a tek `PUT` girişi eklendi. Değişen alanlar: `claim`, `provenance.experiment_id`/`data_time_range`, `effect_size` (artık `avg_net_bps=32.47`/`wr=0.624` = **cycle-düzeyi** birincil metrik; `cycle_n=93` vb. + `event_n=127` ikincil/etiketli), `scope.coverage_note`, `assumptions` (+1), `confidence.cycle_adjustment_classification`, `evidence_families` (+2), `version` 1→2 (`touch_version()` ile `history`'e eski durum eklendi, `forward_events` 0→0 kaldı — anayasal olarak doğru: materyal değişiklik ileri-kanıtı sıfırlar). **`status`/`permitted`/`forbidden`/`frozen` DEĞİŞMEDİ** (hâlâ FORWARD_VALIDATING, SIZING_ALLOWED hâlâ yasak — terfi YOK). `effect_size` runtime'da yalnız `ami/knowledge/objects.py`'nin kendisi tarafından okunuyor (grep doğrulandı) → bu düzeltme hiçbir strateji/route/eşik davranışını ETKİLEMEDİ.

**Kök-neden derinleştirme — Temmuz olayıyla kesin ayrım kanıtlandı:** `bd7feb32`'nin kendi mesajı: `BINANCE_WS` **2026-07-03**'te (`5cda3122`) `/market/stream`'den (doğru) `/stream`'e (yanlış, "sıfır frame") REGRESE OLDU, 2026-07-06'da restart'ta canlıya geçti, 4 gün sonra tespit edilip 07-10'da düzeltildi. **Bu commit May-Haziran penceresinden SONRA** — `git log` bu pencerede (2026-04-20→06-10) `data/microstructure_collector.py`'ye dokunan SIFIR commit gösteriyor; `BINANCE_WS` boşluk boyunca zaten DOĞRU (`/market/stream`) değerdeydi. **Sonuç: Mayıs boşluğu ile Temmuz olayı KANITLANMIŞ ŞEKİLDE FARKLI mekanizmalar** (kod regresyonu Mayıs'ta yok). Ayrıca `reports/RECONNECTION_AUDIT.md` (2026-03-04 tarihli, ayrı/önceki bir olay) zaten "ISP intermittently blocks Binance WS → VPN/SOCKS5 path" senaryosunu öngörüyor — bu ortamda tekrarlayan bir transport-katmanı risk sınıfı olduğunu doğruluyor.
**[SUPERSEDE — §111'e bakın, 2026-07-11]:** Bu paragrafın altı çizili iddiası ("git log sıfır commit gösteriyor → BINANCE_WS boşluk boyunca zaten DOĞRUYDU" ve ondan türeyen "KANITLANMIŞ ŞEKİLDE FARKLI mekanizmalar" sonucu) **iki bağımsız salt-okunur incelemeyle FAKTÜEL OLARAK YANLIŞ bulundu** ve §111'de düzeltildi. Mantık hatası: "dosyaya dokunan commit yok" → "değer zaten doğruydu" çıkarımı geçersizdir; doğrusu tam tersidir (değer yanlış kalarak pencereyi değişmeden geçti). Bu satır arşiv/kanıt amacıyla YERİNDE bırakılıyor, silinmiyor — kanonik olarak §111'i esas alın.

**Kök-neden güven sınıflandırması: `ROOT_CAUSE_PROBABLE`** (IDENTIFIED değil). Kanıtlanan: (1) gözlenen arıza mekanizması — bağlantı+abonelik başarılı ama sıfır frame; (2) mekanizmanın boşluk SIRASINDA var olduğu (2026-05-03 first-party diagnostik, kontrol stream'leri dahil hepsi DEAD); (3) kod-seviyesi alternatif nedenler KANITLA ELENDİ (endpoint doğruydu, hiç commit yok). **Eksik/kapatılamayan:** (a) kurtarma anında (06-05/06-06) neyin özel olarak değiştiğine dair hiçbir commit/işletmen notu/ağ-yapılandırma kaydı yok — yalnız zamansal çakışan collector restart'ları var, nedensellik kanıtlanmadı; (b) dönemin OS/VPN/router seviyesi logları hiç tutulmamış, geriye dönük incelenemez; (c) `matrix_vpnfree/` testi sonuçsuz (birebir aynı/tekrar dosya).
**[SUPERSEDE — §111'e bakın]:** madde (3)'ün "endpoint doğruydu" alt-iddiası da yanlıştı (§111); ancak bu maddenin asıl KANITLA-ELEME sonucu (bağlantı/transport-seviyesi bir arızanın tüm boşluğu açıklayamayacağı, çünkü mark/agg AYNI soket üzerinde kesintisiz aktı) GEÇERLİLİĞİNİ KORUYOR — yalnız gerekçe cümlesi (endpoint değeri) yanlıştı, sonuç (connection-seviyesi değil) doğru kaldı.

**Verdict: `HOUR17_KO_CORRECTED_ROOT_CAUSE_PROBABLE`.** Terfi YOK, route donmuş kalıyor, hiçbir runtime/DB/checkpoint/PID dokunulmadı. Push YAPILMADI. **(§111'de kategori daraltıldı ve düzeltildi — bu verdict'in "farklı mekanizmalar" gerekçesi artık supersede edildi, sonucun kendisi -terfi yok, route donuk- değişmedi.)**

## 106. RECOMPUTE DALGASI — 4 KO CYCLE-ADJUSTED DÜZELTMESİ + OD-010 STRAY DOC TEMİZLİĞİ (2026-07-11, Opus 4.8)

CT-005/OD-011'in kalan 7 KO'su HOUR17 metodolojisiyle (read-only reprodüksiyon + canonical-v1 cycle collapse) değerlendirildi; 4'ü operatör yetkisiyle `data/ami/knowledge.sqlite`'ta düzeltildi (kanonik `KnowledgeStore.put()`/`touch_version()`, el yazması SQL yok). **Terfi/route/eşik/runtime DEĞİŞMEDİ; hiçbir status/permitted/forbidden alanı genişletilmedi** (4'ü de hâlâ HOLDOUT_VALIDATED, RESEARCH/SHADOW-only, LIVE/SIZING yasak). event-population KO'ları `mechanism_store.sqlite` gated substrate'inden (deflasyon 0.598); MONDAY-VETO HOUR17 silence-predictor popülasyonundan (deflasyon 0.732). Event-level parity 4'ünde de birebir doğrulandı.

| KO | v | Event-level (parity ✓) | Cycle-level | Sınıflandırma |
|---|---|---|---|---|
| MONDAY-VETO | 1→2 | Monday −47.4bps | Pazartesi **N=15** WR26.7% −49.9bps · non-Mon N=78 WR69.2% +48.3 | `DESCRIPTIVE_ONLY_LOW_SAMPLE` (N<20; veto yönü destekli, tam-valide değil); yanlış BLOCKED_BY_SOURCE kaldırıldı |
| BOOK-PULL | 1→2 | favN=58 WR70.7% delta+70.2 | bk_pull-hi **N=32** WR68.8% +53.9 · lo N=26 −26.4 · **delta +80.3** (iki kol≥20) | `CONFIRMED_DIRECTION_HISTORICAL` (cycle'da GÜÇLENDİ) |
| FUNDING-LEVEL | 1→2 | ungated +115 spread | funding-lo **N=27** WR66.7% +54.9 · hi N=31 −14.3 · **delta +69.2** (iki kol≥20) | `CONFIRMED_DIRECTION_MAGNITUDE_REDUCED`; funding kaynağı=`mark_prices.funding_rate` (dolu, bloke DEĞİL) netleştirildi |
| MECH-COMPOSITE | 1→2 | mscore≥4 TEST **N=23** WR82.6% +88 (cherry-pick) | mscore≥4 **N=38** WR68.4% +39.3 (med+37, tot+1492) | `WEAKENED`; N=23 headline superseded-subset olarak korundu (silinmedi) |

**Değişmeyen 3 KO (mutasyon yok, bağımsız-doğrulanmış factual hata yok):** MGMT-6H (OPERATIONAL karşılaştırma sınıfı — sıralama yönü cycle-deflasyondan etkilenmez), SCALEIN-100 (zaten PRELIMINARY, aynı sınıf), PRECASCADE (zaten negatif/navigasyon-only — disposition değişmez). **Recompute dalgası TAMAMLANDI** (8/8 değerlendirildi: 5 KO-düzeltildi [HOUR17+4], 3 no-change).

**DB doğrulama:** sha256 `24db6329…27760` → `095d9c4e…9be04`. `integrity_check`=ok, şema/`user_version` DEĞİŞMEDİ. **11 satırın tam olarak 4'ü değişti, 7'si payload-hash birebir SABİT.** audit_log'a tam 4 yeni PUT (actor `claude-opus-4.8-sevenko-cycle-correction`). Her KO version 1→2 (`touch_version`: eski durum history'e, forward_events=0 reset — anayasal). Çapraz-kesen: 40-günlük Mayıs likidasyon boşluğu (§104/105) tüm gated-KO'ları etkiliyor, `coverage_note` olarak her birine işlendi; eksik-dönem outcome'u impute EDİLMEDİ.

**OD-010 stray-doc:** `AMI_COMMERCE_INTELLIGENCE_OS_IMPLEMENTATION_WHITEPAPER_v1.1_COMPLETE.md` (f02b7d88'de eklenmişti) — kendi header'ı canonical repo target'ı `D:\commerce_intelligence` diyor; hiçbir Eclipse/S34 dokümanı adıyla referans vermiyor; Eclipse governance zincirini kırmıyor; kanonik kopya `D:\commerce_intelligence\docs\architecture\`'da MEVCUT. Stray onaylandı → Eclipse tracking'inden `git rm` ile çıkarıldı (ayrı commit); commerce repo kopyası DOKUNULMADI.

**Verdict: `SEVEN_KO_GOVERNANCE_CORRECTED_OD010_CLEAN`.** Push YAPILMADI.

---

## 107. LIQUIDATION SILENCE / TRANSPORT-OUTAGE DETECTOR — IMPLEMENTED, AWAITING REVIEW (2026-07-11, Sonnet 5)

Confirmed 40-gün-3-saat (2026-04-27T14:24Z→2026-06-06T17:47Z) all-tracked-symbol
likidasyon boşluğu hiçbir zaman kararlı bir canonical health failure olarak
yüzeye çıkmamıştı (bkz. §104/105). Yeni, tamamen izole (mevcut hiçbir dosya
DEĞİŞTİRİLMEDİ — `tools/native_ws_health_policy.py` dahil, o başka bir
session'ın kirli/uncommitted dosyası, yalnız referans için okundu) detector
eklendi: `tools/liquidation_silence_policy.py` (saf karar politikası,
per-symbol + all-tracked-symbol + control-stream cross-validation) +
`tools/liquidation_silence_detector.py` (bounded read-only snapshot + one-shot
CLI, `tools.health_state.write_component_health` üzerinden yalnız kendi
`logs/health/liquidation_silence.json` dosyasını yazar — `overall.json`'a
ASLA yazmaz, `tools/heartbeat_watchdog.py` DEĞİŞTİRİLMEDİ, disabled-by-default).

**Kalibrasyon (salt-okunur, `data/microstructure.db` mode=ro):** post-
2026-06-06 (mevcut `all_market_arr` mimarisi) 3 sağlıklı pencere (~19 gün,
Temmuz 6-10 outage hariç) → all-tracked-symbol simultaneous-silence max
gözlenen=2508.9s. Donmuş eşikler: `ALL_SYMBOL_SILENCE_WARNING=3600s` (1.43x
marj), `ALL_SYMBOL_SILENCE_CRITICAL=7200s` (2.87x marj), `SYMBOL_SILENCE_
WARNING=9000s` (per-symbol, hiçbir zaman tek başına RED üretmez),
`CONTROL_STREAM_FRESH=300s`. Policy fingerprint (sha256):
`9781e0ed8f7b4950e62bdb6b4e64773ef1f9f6e383749b92ac20641dec4ed9d8`.

**Test:** 52/52 passed (`test_liquidation_silence_policy.py` 28 +
`test_liquidation_silence_detector.py` 24 collected, `--basetemp` scratchpad,
`-p no:cacheprovider`). Gerçek prod DB üzerinde iki tarihsel replay:
Nisan-Haziran boşluğu → `LIQUIDATION_TRANSPORT_OUTAGE`/RED (latency ≈2.3h vs.
gerçekte 40 gün); Temmuz 6-10 routed-endpoint olayı → aynı sınıflandırma.
310 sağlıklı-dönem saatlik replay → 0 false positive. Recovery-to-GREEN
latency ≈5-10s (freshest-of-3 tasarımı sayesinde).

**Bulunan yan-hata (bu batch'te düzeltildi):** ilk taslakta `now_ts` üst
sınırı yoktu → tarihsel replay gerçek duvar saatine sızıyordu (lookahead).
`ts_ms <= now_ts*1000` eklendi (indexed, hâlâ bounded — `EXPLAIN QUERY PLAN`
ile doğrulandı).

**Performans provası:** gerçek 741GiB prod DB'ye karşı tek-atış run
0.0035s, 5 sorgu, hepsi covering-index (full scan yok). Sabit tarihsel
pencere row-count (2026-04-01/02 BTC/ETH/SOL) rehearsal öncesi/sonrası
birebir aynı. `mode=ro` INSERT'i `sqlite3.OperationalError` ile reddediyor
(test-kanıtlı). Runtime/PID/live-executor durumu değişmedi (12 proses,
duplicate yok, live executor yok — önce/sonra birebir).

**Aktivasyon YOK:** `tools/heartbeat_watchdog.py`'nin `OPTIONAL_COMPONENT_
FILES`'ına eklenmedi, `start_eclipse.ps1`'e eklenmedi, hiçbir proses
başlatılmadı/restart edilmedi. `compose_with_overall_severity()` gelecekteki
controlled-activation batch'i için tanımlı+izole test edildi, hiçbir üretim
kod yolundan çağrılmıyor.

Tam rapor: `reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md`.

**Verdict: `LIQUIDATION_SILENCE_DETECTOR_IMPLEMENTED_AWAITING_REVIEW`.** Push YAPILMADI. Next: `REVIEW_LIQUIDATION_SILENCE_DETECTOR`. **(§108'de düzeltildi — bu verdict artık supersede edildi.)**

---

## 108. LIQUIDATION SILENCE DETECTOR — CORRECTIVE IMPLEMENTATION, AWAITING RE-REVIEW (2026-07-11, Opus 4.8)

§107 detector'ı bağımsız (salt-okunur, hiçbir değişiklik yapmayan) review'dan
`LIQUIDATION_SILENCE_DETECTOR_CORRECTIVE_IMPLEMENTATION_REQUIRED` verdict'i aldı
(3 MEDIUM + LOW bulgular). Düzeltmeler uygulandı. **Donmuş eşikler
(3600/7200/9000/300) DEĞİŞMEDİ; detector AKTİVE EDİLMEDİ, schedule EDİLMEDİ,
restart EDİLMEDİ.** Değişen dosyalar yalnız yeni izole detector dosyaları +
rapor + bu governance kaydı (foreign-owned kirli dosyalara DOKUNULMADI).

**MEDIUM 1 — partial-symbol overclaim → complete-evidence gating:** "tüm
izlenen semboller sessiz" iddiası artık HER sembol için kullanılabilir kanıt
gerektiriyor (`complete_symbol_evidence`, `missing_symbols`, known/tracked
count'ları çıktıya eklendi). BTC+ETH critical + SOL missing artık
`LIQUIDATION_TRANSPORT_OUTAGE`/RED DEĞİL → yeni `PARTIAL_SYMBOL_EVIDENCE`/UNKNOWN
(regression testi eklendi). Native-WS RED yukarı-akış RED'i korur; complete
evidence'ta izole sembol uyarısı korunur.

**MEDIUM 2 — fingerprint eksikliği → decision-logic fingerprint:**
`POLICY_FINGERPRINT` artık 4 eşik+versiyon string yerine tam `POLICY_SPEC`
(precedence, boundary operatörleri, aggregation metodu, complete-evidence
şartı, native-WS/collector precedence, future-ts tolerance, output schema...)
üzerinden sha256. Sensitivity testleri: semantik değişince değişir, key
sırasına duyarsız, v2 ≠ v1.
- Eski (superseded, v1): `9781e0ed8f7b4950e62bdb6b4e64773ef1f9f6e383749b92ac20641dec4ed9d8`
- Yeni (v2): `e117cf132bce3bd180af3c718670d3c75910dd69206588d4b7f1b341aadf2291`
- `POLICY_VERSION = liquidation_silence_policy_v2_2026-07-11`.

**MEDIUM 3 — historical replay temporal kontaminasyonu → evaluation mode:**
`evaluate_once`/`run_once` artık explicit `evaluation_mode`
(`LIVE`|`HISTORICAL_REPLAY`) alıyor. LIVE: `now_ts` wall-clock'a yakın olmalı
(≤900s), live default dosyaları okunabilir. HISTORICAL_REPLAY: live default
overall/collector/pid dosyaları OKUNAMAZ (`HISTORICAL_EVIDENCE_REQUIRED`),
component ts `now_ts`'e göre doğrulanır (gelecek ts →
`CONTROL_COMPONENT_TEMPORAL_MISMATCH`). Yapısal API güvencesi, sadece dokümantasyon
uyarısı değil. Tüm replay testleri güvenli moda taşındı.

**Diğer:** structured DB hata kodları (`DB_TABLE_MISSING/SCHEMA_MISMATCH/
LOCKED/READ_ERROR/CONNECT_ERROR/PERMISSION_DENIED`) `error` alanında yüzeye
çıkıyor (sağlıklı koşumda null); future-timestamp handling (≤60s skew clamp,
ötesi anomaly-unusable); symbol normalization (dedup/case/whitespace, boş
evren fail-visible).

**Test:** 86/86 passed (policy 48 + detector 38, `-p no:cacheprovider`,
`--basetemp` scratchpad). Canonical health regression (writer-ownership +
gate-integration) 16/16 passed. Replay revalidation (güvenli HISTORICAL mode):
Nisan onset YELLOW ≈62.6min/RED ≈122.6min; 05-21 control-stale → CONTROL_STREAMS_STALE
(over-claim yok); recovery <1min; Temmuz onset YELLOW ≈64min/RED ≈124min;
sağlıklı 9/9 HEALTHY; dışlanan eski mimari 7 probe HEALTHY (portability kanıtsız).
Performans: LIVE one-shot 0.0043s, 5 sorgu, 1693B izole scratch'e, gerçek
`logs/health/`'e YAZILMADI. Immutability: donmuş tarihsel row-count (BTC/ETH/SOL
2026-04-01..03 = 687/798/0) önce/sonra birebir. 12 python proses, 0 detector,
0 live executor değişmedi.

**Rapor düzeltmeleri:** "~19 gün" → 307 saat included coverage / ≈12.8 elapsed
gün / 14 ayrı takvim günü; "23x" → ≈25.6x; kalibrasyon
`ADEQUATE_WITH_LOW_LIMITATION` (0 FP yalnız değerlendirilen sağlıklı pencerelerde;
dışlanan eski rejimde 1 gerçek YELLOW mümkün, mevcut mimaride false positive
değil ama portability kanıtsız).

Tam düzeltme kaydı: rapor §19.

**Verdict: `LIQUIDATION_SILENCE_DETECTOR_CORRECTED_AWAITING_REREVIEW`.** Push YAPILMADI. Aktivasyon/execution yetkisi YOK. Next: `REREVIEW_LIQUIDATION_SILENCE_DETECTOR`. **(§109'da kabul edildi — bu verdict artık supersede edildi.)**

---

## 109. LIQUIDATION SILENCE DETECTOR — INDEPENDENT RE-REVIEW ACCEPTED (2026-07-11, Sonnet 5)

§108'in düzelttiği implementasyon bağımsız (salt-okunur, hiçbir dosya
değiştirmeyen) re-review'dan geçti. **Final verdict:
`LIQUIDATION_SILENCE_DETECTOR_ACCEPTED`.** Bu kayıt yalnız governance/
implementasyon kabulüdür — **detector devre dışı ve inaktif kalmaya devam
ediyor**; hiçbir aktivasyon, schedule, proses başlatma/restart, push veya
execution yetkisi verilmedi.

**3 MEDIUM bulgunun tamamı kapatıldı (bağımsız doğrulandı):**
1. Partial-symbol evidence overclaim — BTC/ETH critical + SOL missing artık
   `PARTIAL_SYMBOL_EVIDENCE`/UNKNOWN üretiyor, `LIQUIDATION_TRANSPORT_OUTAGE`/RED
   DEĞİL; `ALL_SYMBOL_SILENCE_BEYOND_*` reason kodu kullanılmıyor.
2. Eksik policy fingerprint — `POLICY_FINGERPRINT` artık tam `POLICY_SPEC`
   (precedence/operator/aggregation/complete-evidence/native-WS precedence/
   future-tolerance/severity-map/schema) üzerinden; bağımsız doğrulama:
   `e117cf132bce3bd180af3c718670d3c75910dd69206588d4b7f1b341aadf2291` (v1'den
   farklı, key-sırasına duyarsız, deterministik).
3. Historical replay/live evidence kontaminasyonu — `LIVE`/`HISTORICAL_REPLAY`
   mode ayrımı yapısal olarak zorunlu kılınıyor (live default path'ler
   HISTORICAL modda reddediliyor, component ts `now_ts`'e göre doğrulanıyor);
   bağımsız test edildi, bypass edilmedi.

**Yeni HIGH/MEDIUM bulgu YOK.** 3 LOW gözlem non-blocking olarak kaydedildi
(düzeltme yetkisi VERİLMEDİ, yalnız bilgi amaçlı):
- detector-layer I/O/mode contract manuel schema-version string ile
  yönetiliyor (türetilmiş bir detector-contract fingerprint değil);
- malformed-input validasyonu native-WS RED'den önce geliyor, ama üretim
  detector'ı her zaman well-formed input veriyor ve arıza UNKNOWN olarak
  görünür kalıyor;
- duplicate DB hata kodları de-duplicate edilmiyor (kozmetik).

**Bağımsız test doğrulaması (izole `--basetemp`, `-p no:cacheprovider`):**
policy 48 + detector 38 + canonical health regression (writer-ownership +
gate-integration) 16 = **102 passed, 0 failed, 0 skipped**. Replay sonuçları
ve rapor düzeltmeleri (307h/≈12.8 gün/14 takvim günü, ≈25.6x,
`ADEQUATE_WITH_LOW_LIMITATION`) bağımsız olarak yeniden üretildi ve eşleşti.
Runtime, veritabanları, canonical health çıktıları, PID/checkpoint dosyaları
ve foreign-owned dosyalar DEĞİŞMEDİ (12 Eclipse-Scalper python prosesi, 0
duplicate, 0 detector proses, 0 live executor — önce/sonra birebir; health
`ok`, native WS `GREEN`).

**Kanonik durum geçişi:** `LIQUIDATION_SILENCE_DETECTOR_CORRECTED_AWAITING_
REREVIEW` → `LIQUIDATION_SILENCE_DETECTOR_ACCEPTED`. Eşikler (3600/7200/9000/
300) DEĞİŞMEDİ. Hiçbir runtime dosyası oluşturulmadı. `OPTIONAL_COMPONENT_
FILES`'a ve `compose_with_overall_severity`'ye wiring HALEN ERTELENMİŞ — ayrı,
operatör-onaylı bir controlled-activation kickoff'u gerektiriyor.

Tam bağımsız re-review raporu: bu oturumun re-review mesajında (aynı sohbet
geçmişinde) kayıtlı; rapor dosyasına ayrıca yazılmadı (mevcut MD-update
politikası: batch sonunda yalnız SYSTEM_STATE + PROGRESS_LEDGER güncellenir,
yeni MD dosyası istisnadır).

**Verdict: `LIQUIDATION_SILENCE_DETECTOR_ACCEPTED`.** Push YAPILMADI. Aktivasyon/
schedule/restart/execution yetkisi YOK. Next: `AWAIT_EXPLICIT_CONTROLLED_
ACTIVATION_KICKOFF`.

---

## 110. GOVERNANCE İLKESİ — KADEMELİ BAĞIMSIZ İNCELEME ZİNCİRİ (ZORUNLU, 2026-07-11)

Operatör kararı: doğrulama gerektiren her önemli implementasyon/düzeltme
**standart, sıkıştırılamaz bir kademe zincirinden** geçer:

`implementation → bağımsız review → correction → bağımsız re-review → acceptance`

- Fazlar **tek tek ve ayrı**; aralarında **operatör sign-off** var. İki faz
  (özellikle implementation + review) asla aynı geçişte birleştirilmez.
- **Gerekçe:** review'ın epistemik değeri bağımsızlığında. Aynı aktör bir
  artefaktı tek geçişte hem üretir hem onaylarsa = self-approval = değersiz.
  Her review **salt-okunur**, hiçbir dosyayı değiştirmez; düzeltme ayrı
  `correction` fazında.
- **Otomatik ama kapılı:** disiplin varsayılan olarak kendiliğinden uygulanır;
  insan kapısı (sign-off) kaldırılmaz. Zaman baskısı sıkıştırma gerekçesi
  değildir.
- Her faz kendi verdict token'ıyla kapanır ve buraya kaydedilir.
- **Kanonik emsal:** liquidation-silence detector zinciri §107 (implemented→
  awaiting review) → bağımsız review (CORRECTIVE_REQUIRED) → §108 (corrected→
  awaiting re-review) → bağımsız re-review (ACCEPTED) → §109 (acceptance
  kaydı). Aynı desen bundan sonra tüm doğrulama-gerektiren işlerde geçerli.
- Bu ilke `CLAUDE.md` "Kademeli bağımsız inceleme zinciri" bölümüne de işlendi
  (her oturum otomatik yüklenir) ve kalıcı memory'ye
  `feedback_gated_independent_review_chain` olarak yazıldı.

Bu governance-only kayıttır; hiçbir kod/runtime değişmedi. Push YAPILMADI.

---

## 111. MAYIS 2026 LİKİDASYON BOŞLUĞU — GİT-TARİHİ DÜZELTMESİ + KATEGORİ DARALTMA (2026-07-11, Opus 4.8)

**Bu bölüm §104/§105'teki bir FAKTÜEL HATAYI düzeltir.** İki bağımsız salt-okunur
inceleme (araştırma + adversarial çürütme denemesi, ayrı temiz-context ajanlar,
[[feedback_gated_independent_review_chain]] disiplinine uygun — implementation→
review yerine burada araştırma→adversarial-verify) aynı sonuca ulaştı: §105'in
"BINANCE_WS boşluk boyunca zaten DOĞRU (/market/stream) değerdeydi → Mayıs ile
Temmuz KANITLANMIŞ ŞEKİLDE FARKLI mekanizmalar" iddiası **yanlış**. Bu bölüm
kendisi de bağımsız bir re-review bekliyor (aşağıya bakın, henüz `ACCEPTED`
değil) — [[feedback_gated_independent_review_chain]] gereği ben (orkestratör)
düzeltmeyi yazdım, kabul kararını ayrı bir taze-context ajan verecek.

**Faz 1 (araştırma, salt-okunur):** TRANSPORT_SPECIFIC_OUTAGE kategorisinin
altında daha spesifik bir mekanizma önerdi: gap-era collector
(`data/microstructure_collector.py`@`dc92b9b0` `_build_stream_url` satır
453-458) forceOrder+aggTrade+markPrice'ı **tek soket**te çoğulluyordu; DB
(mode=ro, bounded) 40 gün boyunca **liq=0 iken mark/agg kesintisiz aktığını**
gösterdi (aynı soket) → neden connection/transport-seviyesi DEĞİL, yalnız
`forceOrder` alt-akışına özgü. Önerilen kategori: **`LIQUIDATION_FORCEORDER_
SUBSTREAM_SPECIFIC_SILENCE_ON_HEALTHY_SOCKET`**.

**Faz 2 (adversarial çürütme denemesi, null hipotez = "mevcut §104/105 doğru"):**
4 iddianın hiçbiri çürütülemedi; DB sınırları (`14:27:26.345Z`/`17:43:52.123Z`)
milisaniyeye kadar bağımsız yeniden doğrulandı. **Ayrıca §105'in kendi iddiası
git-arkeolojisiyle YANLIŞLANDI:**
- `git log -S'market/stream' --all` (tüm dal+stash) → `/market/stream` string'i
  git tarihçesine **yalnızca `bd7feb32` (2026-07-10)** ile giriyor.
- `dc92b9b0`(2026-03-05) → `5cda3122`(2026-07-03) arası **her commit'te**
  `BINANCE_WS = "wss://fstream.binance.com/stream"` (yönlendirilmemiş) —
  Mayıs-Haziran penceresi dahil hiç değişmeden kalmış.
- `reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-06-06.md` git'e
  **hiç eklenmemiş** (untracked) — 06-06 kurtarmasını anlatan tek belge.
- **Mantık hatası teşhisi:** "dosyaya dokunan commit yok" → "değer zaten
  doğruydu" çıkarımı geçersiz. Doğru çıkarım: değer (`dc92b9b0`'da zaten
  yanlış olan `/stream`) pencere boyunca **değişmeden, yanlış kalarak**
  geçti — `5cda3122`(07-03)'ün "corrected /market/stream → /stream" commit
  mesajı bunu ayrıca doğruluyor (mesaj "düzeltilmiş"ten "bozulmuş"a
  regresyon anlatıyor, ki bu ancak önceki değer zaten `/stream` DEĞİLSE
  tutarlı olur — burada mesaj/diff arasında da bir tutarsızlık var, ek
  inceleme notu).

**Düzeltilmiş sonuç:** May-Temmuz ayrımının doğru gerekçesi **endpoint değeri
değil, KAPSAM**dır — Mayıs: sağlıklı soket üzerinde alt-akış-seçici (yalnız
`forceOrder`) sessizlik; Temmuz: kod regresyonuyla (`5cda3122`) tüm-soket
sıfır-frame (mark/agg yalnız 07-03'te eklenen REST fallback sayesinde
kurtuldu — Mayıs'ta bu fallback henüz yoktu). §105 madde (3)'ün "connection-
seviyesi bir arıza tüm boşluğu açıklayamaz" SONUCU geçerliliğini koruyor
(mark/agg aynı soket üzerinde kesintisiz aktı) — yalnız gerekçe cümlesi
("endpoint doğruydu") yanlıştı.

**Kategori güncellemesi:** `TRANSPORT_SPECIFIC_OUTAGE` → **`LIQUIDATION_
FORCEORDER_SUBSTREAM_SPECIFIC_SILENCE_ON_HEALTHY_SOCKET`** (dar, kanıtlı).
`ROOT_CAUSE_PROBABLE` sınıflandırması KORUNUYOR (değişmedi) — yalnız
destekleyici gerekçe düzeltildi, güven seviyesi düşmedi çünkü asıl kanıt
(DB + kod) sağlam kaldı, yalnızca bir yardımcı git-iddiası yanlıştı.

**Kapatılamayan (disiplinle korunuyor, yeniden açılmadı):** Binance'in
sembol-bazlı `forceOrder`'ı sunucu tarafında neden 40 gün sustuğu yerel
kanıttan KANITLANAMAZ (dönem stdout logu saklanmamış) — iki ajan da bunu
spekülasyona açmadı, aynı disiplin burada da korunuyor.

**Değişmeyen:** Terfi YOK, route donmuş kalıyor, hiçbir kod/DB/runtime/PID
dokunulmadı — bu salt-okunur bir governance-metni düzeltmesidir.
`reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_
FORENSIC_2026-07-11.md`'nin ilgili paragrafına (satır 66) da aynı
supersede-yerinde-bırakma + düzeltme eklendi (rapor §"Correction Addendum").

**Verdict: `MAY_GAP_ROOT_CAUSE_CORRECTED_AWAITING_INDEPENDENT_REREVIEW`.**
Push YAPILMADI. Kabul kararı bekliyor — bağımsız taze-context bir ajan bu
düzeltmeyi kanıta karşı denetlemeden `ACCEPTED` denmeyecek. Next:
`REREVIEW_MAY_GAP_ROOT_CAUSE_CORRECTION`. **(§112'de kabul edildi.)**

---

## 112. MAYIS 2026 KÖK-NEDEN DÜZELTMESİ — BAĞIMSIZ RE-REVIEW KABUL EDİLDİ (2026-07-11, Opus 4.8)

§111'in düzeltmesi bağımsız (salt-okunur, hiçbir dosya değiştirmeyen, taze-
context) bir re-review'dan geçti. **Final verdict: `ACCEPTED`.**

Reviewer önceki hiçbir ajanın bulgusuna güvenmeden kendi başına yeniden
üretti: `git log -S'market/stream' --all` → yalnız `bd7feb32`; `dc92b9b0` ve
`5cda3122`'nin ikisinde de `BINANCE_WS="/stream"` (aynı, değişmemiş) —
bağımsız doğrulandı. Ayrıca kendi başına yeni bir tutarsızlık da buldu:
`5cda3122`'nin commit mesajı bir URL düzeltmesi anlatıyor ama diff'i
`BINANCE_WS` satırını hiç değiştirmiyor (yalnız context) — bu, düzeltmenin
zaten ölçülü şekilde not düştüğü noktayla örtüşüyor. DB kanıtı (05-15, 05-25)
bağımsız sorguyla doğrulandı. Overclaim kontrolü: aşırı iddia yok, dil ölçülü,
`ROOT_CAUSE_PROBABLE` korunmuş, "Binance neden sustu" sınırı yeniden
açılmamış. Supersede-in-place disiplini doğrulandı (orijinal yanlış metin
silinmemiş). Kapsam yalnız 2 dosya (`SYSTEM_STATE.md` + rapor). §104/105/111
"route donmuş" ifadesi tutarlı.

**Kanonik durum geçişi:** `MAY_GAP_ROOT_CAUSE_CORRECTED_AWAITING_INDEPENDENT_
REREVIEW` → **`MAY_GAP_ROOT_CAUSE_CORRECTION_ACCEPTED`**. Bu yalnız
governance/bulgu kabulüdür — kategori adı (`LIQUIDATION_FORCEORDER_
SUBSTREAM_SPECIFIC_SILENCE_ON_HEALTHY_SOCKET`) kanonik kabul edildi, ancak
hiçbir kod/route/eşik/terfi değişmedi; route hâlâ donmuş. Binance'in sunucu
tarafı sessizlik nedeni hâlâ kanıtlanamaz durumda, yeniden açılmadı.

**Verdict: `MAY_GAP_ROOT_CAUSE_CORRECTION_ACCEPTED`.** Push YAPILMADI.
Terfi/route değişikliği YOK. Next: bekleyen aksiyon yok, bulgu kanonik kayıtta.

---

**Operasyonel not (2026-07-11, ~10:41:45Z):** Bu batch sırasında
`tools/heartbeat_watchdog.py` prosesi (eski PID 22816) durduğu tespit edildi
— OS-seviyesi crash kaydı yok (Windows Event Log'da python.exe için fault
kaydı yok), kesin kök neden belirlenemedi. Veri katmanı (collector/
bookticker/oi_poller/event_diary/shadow runner'lar, 11/11 PID birebir
baseline'la eşleşiyor, 0 duplicate) TAMAMEN SAĞLAM ve canlı yazmaya devam
ediyor (`collector.json`/`bookticker.json` kendi sahiplerince bağımsız
güncelleniyor). Yalnız agregatör/watchdog durdu — `overall.json`/
`watchdog.json` ~10:41:45Z'den beri donuk. **Restart operatörün işi**
(`start_eclipse.ps1`, sandbox dışında) — bu batch hiçbir restart komutu
çalıştırmadı, çalıştırmayacak. Bu governance kaydından sonra bu oturum yeni
bir arka-plan işi başlatmıyor; operatör watchdog'u ele alana kadar bekliyor.
**(§113'te operatör yetkisiyle push + kontrollü watchdog başlatma yapıldı.)**

---

## 113. OPERASYONEL AKTİVASYON — PUSH + KONTROLLÜ WATCHDOG BAŞLATMA (2026-07-11, Opus 4.8)

Operatörün açık yetkisiyle (`PUSH_ACCEPTED_COMMITS_AND_START_WATCHDOG`) push
ve kontrollü watchdog başlatma yapıldı.

**Push (başarılı, temiz fast-forward):** 3 kabul edilmiş commit
`origin/codex/data-layer-fallback-cleanup`'a push edildi:
`0e56c123` (May-gap düzeltmesi), `47f7a205` (detector activation wire),
`1b5fec33` (May-gap kabul + watchdog-down notu). `190b064d..1b5fec33`,
force YOK. Remote tip == local HEAD (`1b5fec33`), ahead/behind 0/0, üç commit
de remote'ta mevcut. `main`/`origin/main` DOKUNULMADI. Foreign-owned kirli
7 dosya (native_ws_health_policy.py, s34_cascade_navigation_dashboard.py,
status/stop_eclipse.ps1, test_native_ws_health_policy.py,
.claude/settings.local.json, runtime/dashboard_backend.json) hiçbir push
commit'ine girmedi, dokunulmadı.

**Watchdog başlatma (`start_eclipse.ps1 -NoCleanStop`):** `-NoCleanStop`
seçildi — full-stack bounce YOK; her rol idempotent (çalışan prosesi
PID-file/command-needle ile tespit edip `already_running` işaretliyor),
yalnız EKSİK olan `heartbeat_watchdog` başlatıldı. `-EnableLive` YOK →
live executor aktif olarak KAPALI tutuldu (pid=0). Yeni watchdog PID=12652.
Önceki 11 sağlıklı veri-katmanı prosesi restart EDİLMEDİ (birebir aynı
PID'ler korundu).

**Post-start doğrulama (hepsi geçti):** tam olarak 1 watchdog, 12 Eclipse
prosesi, 0 duplicate, 0 live executor; `overall.json` yeniden canlı
(10:41'den beri donuktu, artık ~5s'de bir yazılıyor); mevcut componentler
(bookticker/collector/paper_trader/watchdog) korundu, düşmedi; native WS
GREEN; canonical health ok/GREEN.

**Detector çıktısı (canonical path, item 6/7):** detector'ın kabul edilmiş
**tek-atış** (`run_once`, LIVE) invocation'ı çalıştırıldı →
`logs/health/liquidation_silence.json` yazıldı. Doğrulama: status=HEALTHY,
severity=GREEN, schema=`liquidation_silence_component_v2`, policy=v2,
**fingerprint `e117cf132bce3bd180af3c718670d3c75910dd69206588d4b7f1b341aadf2291`**
(kabul edilenle birebir), `error=null`, eval_mode=LIVE, tracked semboller
normalize (`canonical_runtime_config`), `complete_symbol_evidence=True`,
partial-evidence altında RED YOK. Çalışan watchdog bir sonraki cycle'da bunu
okudu → `components[]`'e `liquidation_silence` (GREEN) eklendi ve
`compose_with_overall_severity` fold'u overall'ı GREEN bıraktı (GREEN=no-op,
tasarım gereği). RED→halted / YELLOW→degraded dalları merge edilmiş test
suite'inde (64 passed) kanıtlı; native_ws RED precedence'i fold sırası + compose
never-downgrade ile korunuyor.

**Soak (~55s, çoklu cycle):** watchdog canlı, PID stabil (12652), 0 duplicate,
component timestamp'leri ilerliyor (11:43:22→11:44:10), detector runtime
hafif (2.1ms), stderr temiz (0 istisna), dosya büyümesi kontrollü, beklenmedik
YELLOW/RED geçişi yok, canonical-health writer çakışması yok, native WS GREEN,
live executor OFF.

**DB immutability:** detector/watchdog mode=ro; donmuş tarihsel pencere
(2026-04-01..03 BTC/ETH/SOL liq = 687/798/0) push+start+soak öncesi/sonrası
birebir aynı. Collector'lar canlı veriyi meşru şekilde ekliyor (ayrı).

**⚠️ ÖNEMLİ DÜRÜSTLÜK NOTU — sürekli izleme DEĞİL:** detector tasarım gereği
disabled-by-default **tek-atış**; ne `start_eclipse.ps1` ne bir scheduler onu
periyodik çalıştırıyor. Yani `liquidation_silence.json` şu an **tek bir
anlık-görüntü** (mtime 11:41:48'de donuk, kendini yenilemiyor). Watchdog bunu
her cycle okuyor ama içerik GREEN olduğu için fold no-op (zararsız). **Sürekli
liquidation-silence izlemesi için ayrı bir scheduling batch'i (cron/loop)
gerekir** — bu ayrı bir operatör kararıdır, bu batch'in kapsamı dışındadır. Bu
batch yalnız READ-path'i canlıya aldı + tek doğrulama anlık-görüntüsü üretti.

**Verdict: `WATCHDOG_AND_LIQUIDATION_DETECTOR_OPERATIONALLY_ACTIVATED`**
(wire operasyonel + tek-atış çıktı doğrulandı; periyodik scheduling ayrı,
ertelenmiş). Live executor OFF, hiçbir trade/order/execution yolu aktive
edilmedi, DB mutasyonu yok, foreign dosyalar dokunulmadı. Governance commit'i
push edildi.

---

## 114. AÇIK OPERASYONEL TAKİP MADDELERİ — İSİMLENDİRİLMİŞ, AÇIK, ENGELLEYİCİ DEĞİL (2026-07-11, Opus 4.8)

§113 sonrası iki açık operasyonel takip maddesi bilinçli olarak **açık**
bırakılıyor — **sessizce kapatılmış SAYILMAZLAR**. İkisi de isimlendirilmiş,
engelleyici değil, operatör kararı veya yeni kanıt bekliyor. Bu governance-only
kayıttır; hiçbir kod/runtime/scheduler/DB değişmedi, hiçbir şey başlatılmadı.

### 1. `DETECTOR_SCHEDULING_DECISION_PENDING`
- Liquidation-silence entegrasyon wire'ı **operasyonel ve kabul edilmiş**
  (§107-113).
- Geçerli bir **tek-atış** component çıktısı üretildi ve canonical health'e
  fold edildi (§113 — GREEN no-op, fingerprint `e117cf13…`, error=null).
- Detector **disabled-by-default** ve şu an **schedule EDİLMEMİŞ** (ne
  `start_eclipse.ps1` ne scheduled-task ne recurring loop; doğrulandı).
- Dolayısıyla `logs/health/liquidation_silence.json` **statik bir tek-atış
  anlık-görüntüsüdür**, sürekli izleme DEĞİLDİR (kendini yenilemiyor).
- Bu **kasıtlı bir controlled-activation sınırıdır** — "sürekli izleme aktif"
  iddiası DEĞİLDİR.
- Periyodik yürütme, scheduling sıklığı, stale-output politikası, restart
  davranışı, soak gereksinimleri ve rollback kriterleri **ayrı, açık bir
  operatör-onaylı batch** gerektirir.
- **Bu governance kaydı hiçbir scheduling veya ek yürütme yetkisi VERMEZ.**

### 2. `WATCHDOG_SILENT_STOP_ROOT_CAUSE_OPEN`
- Önceki watchdog prosesi (eski PID 22816) **eksik bulundu** ve kabul edilmiş
  başlatma yolu (`start_eclipse.ps1 -NoCleanStop`) ile başarıyla **geri
  yüklendi** (yeni PID 12652, §113).
- **Kesin durma nedeni belirsiz** — bir OS crash kaydı veya yeterli nedensellik
  kanıtı bulunamadı.
- Durma **spesifik bir nedene ATFEDİLMEZ** (kanıt yok; spekülasyon üretilmiyor).
- Soru **açık ama inaktif** kalıyor.
- Yeni bir adli inceleme YALNIZCA şu durumlarda açılmalı: (a) sessiz-durma
  deseni tekrarlarsa; (b) ikinci bağımsız veri noktası çıkarsa; veya (c) yeni
  log / OS kanıtı / exit code / proses telemetrisi erişilebilir olursa.
- Mevcut watchdog **sağlıklı**; **acil düzeltici implementasyon yetkisi
  VERİLMEMİŞTİR.**

**Her iki madde de:** açık · isimlendirilmiş · engelleyici değil · kasıtlı
olarak operatör kararı/yeni kanıt bekliyor · sessizce kapatılmış değil.

**Verdict: `PENDING_OPERATIONAL_FOLLOWUPS_RECORDED`.** Push edilecek.
Scheduling veya watchdog adli incelemesi bu batch'te BAŞLATILMADI. Next:
`AWAIT_OPERATOR_DECISION_OR_NEW_WATCHDOG_EVIDENCE`.

---

## 115. PARALEL DEVAM — 3 TRACK KOORDİNASYONU (2026-07-11, Opus 4.8)

Operatör yetkisiyle (`PARALLEL_CONTINUATION_...`) kalan 3 track eşzamanlı,
salt-okunur preflight/forensic ajanlarıyla incelendi (base `16ab3a1e`). Sonuç:
yalnız **1** track'te uygulanabilir iş çıktı (Track 2). Track 1/3 kanonik
devam noktalarında, uygulanabilir iş YOK. Hiçbir production runtime mutasyonu
yapılmadı; watchdog=1, 12 proses, 0 duplicate, live executor OFF, health
ok/GREEN, native WS GREEN — tüm batch boyunca korundu. **Governance entegrasyonu
seri**; her track ayrı bölüm (§115 bu koordinasyon, §116 Track1, §117 Track3;
Track2 kabulü aşağıda).

### TRACK 2 — Storage range-read consumer migration V9 — **ACCEPTED**
Read-path-only, kabul edilmiş V1-V8 serisinin devamı. 3 clean non-mixed
consumer (`research_s34_500k_daytrend_route_sweep.py`,
`research_s34_trailing_oos_realfill.py`, `research_s34_early_confirmation_scan.py`)
full-window `mark_prices` range-read'i `ami.storage.research_reader`'a taşındı;
eski doğrudan-SQL `path_marks()` bit-identical parity oracle olarak korundu
(`path_marks_v2` reader ikizi, boundary `end_ms=hi+1` ile oracle inclusive-upper
≡ reader half-open eşlemesi). Her consumer için `*_reader_migration_parity`
testi eklendi; inventory JSON committed scanner/classifier ile tazelendi
(MIGRATED 21→24, REMAINING 13→10; ayrıca committed snapshot'ın scope
bayatlığı 320→141 düzeltildi).

**Gated zincir:** izole worktree'de implement → **bağımsız review PASS**
(taze-context, testleri kendi çalıştırdı, boundary'yi `research_reader`
kaynağını okuyarak doğruladı, inventory JSON'u committed scanner'dan 0-diff
reprodüksiyon ederek fabrikasyon olmadığını kanıtladı) → ana ağaca merge →
**combined regression 69 passed / 0 failed / 0 skipped** (gerçek prod DB +
archive Parquet mevcut olduğundan `@requires_real_db` testleri de KOŞTU, yalnız
sentetik fixture değil). Catalog immutability: `entry_count=3`,
`index_self_hash=b2b26d06…` önce/sonra DEĞİŞMEDİ. `ami/storage/*` dokunulmadı,
`data/archives/**` mutasyonu yok, mode=ro, scheduler/purge/VACUUM/collector/live
YOK. Implementasyon commit'i `717d7308`. **Kapsam-dışı (ayrı serialized runtime
gate'ler):** scheduler, purge, VACUUM, dependency release, republication/restart
— hepsi DISABLED kalıyor; `research_dependency_status=BLOCKED` (consumer estate
tam migrate edilene kadar). Push YAPILMADI.

## 116. TRACK 1 — OD-SWEEP + SHADOW/PAPER — SALT-OKUNUR PREFLIGHT (2026-07-11)

Bağımsız salt-okunur forensic (base `16ab3a1e`): OD-sweep + shadow/paper
araştırması **kanonik devam noktasında** — aktivasyon COMPLETE (§103, commit
`74673925`), her iki motor CANLI ve sağlıklı (paper PID 19504, realtime-shadow
PID 21576, doğrulandı), `min_gap_semantics=persistent-v2`, pre-v2 pop=1339
frozen/unpoolable, forward v2 N≈6. Yöneten stopping-rule **zaman-tabanlı**
(OD-005/OD-008: ≥6 ay forward, değerlendirme ~2027-01) → uygulanabilir
implementation, resume edilecek açık numeric-N prereg veya restart edilecek
kesintiye uğramış-geçerli koşum **YOK**; yeni prereg OD-008 ile yasak. Bilimsel
kısıtlar korundu (forward-only, mezarlık reopen yok, observation/paper-only,
live yetkisi yok). Hiçbir dosya yazılmadı, hiçbir proses başlatılmadı, DB
sorgusu çalıştırılmadı (durum committed `S34_SHADOW_PAPER_STATUS.json`'dan
okundu). **Verdict: `FORWARD_EVIDENCE_COLLECTION_REQUIRED`** — pasif forward
gözlem devam eder, aksiyon yok. Runtime koordinasyon notu: Track 1 `book_ticker`
tablosu + bookticker collector üzerinde Track 2 ile RO bağımlılık paylaşıyor
(kaynak-dosya çakışması YOK); Track 2 V9 read-path-only olduğundan canlı paper
fill'leri etkilenmedi.

## 117. TRACK 3 — EPISTEMIC-GATES + CVD — SALT-OKUNUR PREFLIGHT (2026-07-11)

Bağımsız salt-okunur forensic (base `16ab3a1e`): epistemik-gate enforcement
**COMPLETE** — `M-0033` (`51e78673`) mandatory `register_experiment_with_gates`
giriş noktasını wire ediyor (11-adım, tek atomik cross-DB transaction,
fail-closed rollback); `M-0034` (`e8576900`) hem 10-dosya inline-SQL yüzeyini
hem `research.sqlite`/ResearchRegistry projeksiyonunu gate-receipt ile kapatıyor.
§97 ad-hoc `research_s34_*.py` sınıfı gate KAPSAMI DIŞINDA (report+graveyard-only,
`experiment_registry`/`research.sqlite` yazmıyor) — doğru şekilde kaydedilmiş,
kapatılmamış; migration-free izole kapatma yok, deneysel-iddia/ROUTE terfisi
olmadan gereksiz. CVD ailesi `FAM_CVD_WINDOWED_TAKER_FLOW` SCIENTIFICALLY_CLOSED
(NO_RELIABLE_ASSOCIATION, `60c3e26f`); alt-windows forbidden rescue; V2 selektör
`NO_CURRENTLY_ELIGIBLE_INDEPENDENT_FAMILY` (doğru). `FAM_BOOK_SPREAD_DYNAMICS`
LONG **PARKED** (TEST N=18<20; gate floor'u düşürmeyip durdu — registry satırı
yok, nullifier tüketilmedi). Reaktivasyon ≥67 eligible LONG cycle (forward, 9
daha) veya operatör split-ratio kararı gerektirir. Schema 14 + M-0031/0035/0036
ledger sağlam. Bilimsel kısıtlar korundu (nullifier/eligibility bypass yok,
aile uydurulmadı, NO_RELIABLE reopen yok). Hiçbir dosya/DB/migration/proses
dokunulmadı. **Verdict: `NO_CURRENTLY_ELIGIBLE_WORK`** — enforcement batch'i
yok, CVD continuation yok, eligible independent family yok.

**Batch verdict: `PARALLEL_CONTINUATION_TRACK2_V9_ACCEPTED_TRACK1_TRACK3_NO_ELIGIBLE_IMPL`.**
Track 2 V9 kanonik ağaca merge+commit (`717d7308`) edildi + governance kaydedildi;
Track 1 forward-gated, Track 3 no-eligible-work. Push YAPILMADI. Next per track:
Track1=`AWAIT_FORWARD_WINDOW_~2027-01`, Track2=`AWAIT_PUSH_AUTHORIZATION` (sonraki
V10 ayrı batch), Track3=`AWAIT_FORWARD_SAMPLE_OR_OPERATOR_SPLIT_RATIO_DECISION`.
**(V9 push §118'de doğrulandı; V10 sonucu §118'de kaydedildi.)**

---

## 118. STORAGE V9 PUSH + RANGE-READ V10 — NO ELIGIBLE CANDIDATES (2026-07-11, Sonnet 5)

**Faz 1 — V9 push:** Operatör yetkisiyle `717d7308` + `5caead04` commit'leri
`origin/codex/data-layer-fallback-cleanup`'a push edildi (`16ab3a1e..5caead04`,
temiz fast-forward, force yok). Remote tip == local HEAD, 0/0, her iki commit
de remote'ta doğrulandı. `main` dokunulmadı. Foreign-owned kirli dosyalar
(7 dosya) değişmedi, hiçbir commit'e girmedi.

**Faz 2 — V10 preflight (salt-okunur statik sınıflandırma):** Kanonik taban
pushed HEAD `5caead04`. Mevcut kabul edilmiş envanterdeki
**10 `REMAINING_MIGRATABLE_RANGE_READ_CANDIDATE`** tek tek, önceki hiçbir
ret-notuna güvenmeden bağımsız incelendi. Production DB'ye **hiçbir sorgu
çalıştırılmadı** (yalnız statik kod okuma). Hiçbir worktree açılmadı, hiçbir
implementasyon/test değişikliği/merge/runtime aksiyonu/storage mutasyonu
olmadı.

**Sonuç: `STORAGE_RANGE_READ_V10_NO_ELIGIBLE_CANDIDATES`.** V9, sıkı migrasyon
barını (CLEAN_NON_MIXED + tam anlaşılmış semantik + korunmuş bit-identical
oracle + CVD/AMI epistemik çakışma yok + runtime/catalog/archive mutasyonu
yok) sağlayan kalan tüm clean/non-mixed full-window adayları tüketti. Kalan
10 adaydan HİÇBİRİ bu barın tamamını sağlamıyor:

| # | Dosya | Sınıflandırma | Diskalifiye nedeni |
|---|---|---|---|
| 1 | `research_eth_provision_realism.py` | UNSAFE_OR_INELIGIBLE | predicated first-crossing forward-ASOF |
| 2 | `research_nonpredictive_carry_provision.py` | UNSAFE_OR_INELIGIBLE | predicated first-crossing forward-ASOF |
| 3 | `research_s34_btc_microtrend_sweep.py` | MIXED_PARTIAL | non-allowlisted `liquidations` erişimi + CVD/cascade adjacency |
| 4 | `research_s34_buyfade_structural.py` | GOVERNANCE_ONLY | `ami.research.registry`/`ami.constitution` epistemik-gate bağımlılığı |
| 5 | `research_s34_consensus_composite.py` | CVD_ADJACENT | OFI/CVD composite, karışık kanıt |
| 6 | `research_s34_eth_preliq_control.py` | CVD_ADJACENT | eth_preliq avoid-list, shadow-runner, cross-DB bağımlılık |
| 7 | `research_s34_eth_preliq_executable.py` | CVD_ADJACENT | eth_preliq avoid-list, shadow-runner, cross-DB bağımlılık |
| 8 | `research_s34_v6_management_system.py` | UNSAFE_OR_INELIGIBLE | first-crossing + predicate-aggregate; clean full-window yok |
| 9 | `s34_regime_filter_shadow_eval.py` | MIXED_PARTIAL (near-miss) | mark_prices range + book_ticker ASOF; shadow-mirror adjacency; mevcut sıkı kapsamda yetkisiz |
| 10 | `validate_data_research_fitness.py` | MIXED_PARTIAL | gerçek çok-tablolu aggregate semantiği |

**Envanter durumu (değişmedi):** `total_scanned=141`; `MIGRATED_RANGE_READ=24`;
`REMAINING_MIGRATABLE=10`; `NO_OP=83+1`; `DO_NOT_TOUCH=8`;
`BLOCKED_FORWARD_ASOF=7`; `BLOCKED_DIFFERENT_DB=4`; `OUT_OF_SCOPE_UNBOUNDED=3`;
`ASOF_ONLY=1`.

**Kanonik devam durumu:** **`AWAIT_OPERATOR_SCOPE_DECISION_MIXED_PARTIAL_OR_
SHADOW_EVAL`** — sonraki range-read migrasyon ilerlemesi ayrı, açık bir
operatör kapsam-genişletme kararı gerektirir: MIXED_PARTIAL tamamlama
migrasyonları mı kabul edilsin, shadow-evaluation consumer'ları mı dahil
edilsin, range+ASOF karışık consumer'lar mı, çok-tablolu/aggregate
consumer'lar mı? **Bu governance kaydı bu genişletmeyi YETKİLENDİRMEZ.**

**Korunan (değişmedi):** Track 1 `AWAIT_FORWARD_WINDOW_~2027-01`; Track 3
`AWAIT_FORWARD_SAMPLE_OR_OPERATOR_SPLIT_RATIO_DECISION`; detector
`DETECTOR_SCHEDULING_DECISION_PENDING`; watchdog
`WATCHDOG_SILENT_STOP_ROOT_CAUSE_OPEN`.

**Storage runtime gate'leri (DISABLED, bu batch'in kapsamı dışında):**
scheduler aktivasyonu, purge/dependency-release, VACUUM, production rotation,
book_ticker restart/recovery, yıkıcı storage bakımı — hepsi kapalı kalıyor.

Bu governance-only kayıttır; kod/test/envanter/catalog/archive/DB/runtime/
foreign dosya DEĞİŞMEDİ.

**Verdict: `STORAGE_RANGE_READ_V10_NO_ELIGIBLE_CANDIDATES`.** Next:
`AWAIT_OPERATOR_SCOPE_DECISION_MIXED_PARTIAL_OR_SHADOW_EVAL`.
**(§119'da, tek adaya özel dar kapsam kararıyla, kabul edildi ve entegre edildi.)**

---

## 119. SHADOW-EVAL MARK_PRICES RANGE MİGRASYONU — KABUL EDİLDİ + ENTEGRE EDİLDİ (2026-07-11, Opus 4.8 + Sonnet 5)

**Kapsam kararı:** Operatör, `AWAIT_OPERATOR_SCOPE_DECISION_MIXED_PARTIAL_OR_
SHADOW_EVAL`'e yanıt olarak **tek bir adaya özgü, dar bir kapsam genişletmesi**
yetkilendirdi: **`tools/s34_regime_filter_shadow_eval.py`**. Bu, blanket bir
MIXED_PARTIAL/CVD-adjacent/multi-table/aggregate/shadow-consumer yetkisi
DEĞİLDİR ve başka hiçbir adaya otomatik olarak uygulanmaz.

**Aday:** `tools/s34_regime_filter_shadow_eval.py` — yalnız ayrışabilir
`mark_prices` range-read bölümü migrate edildi. Mevcut `book_ticker` ASOF
implementasyonu ve çağrı-site'ı (`book_at_v2`, önceden `b40441f2`/Batch 4'te
migrate edilmişti) **byte-identical korunarak dokunulmadı**.

**Kritik veri-seviyesi coupling korundu:** `simulate_counterfactual()` içinde
range-read'den türeyen `exit_ts_ms`, ASOF çağrısına argüman olarak geçiyor —
yani range-read'deki bir hata ASOF sonucunu da bozabilirdi. Bu yüzden kanıt
yalnız unit-level (`mark_rows` vs `mark_rows_v2`) değil, **uçtan-uca
`simulate_counterfactual()` seviyesinde** de üretildi (tam sonuç sözlüğü +
`exit_ts_ms`/`exit_mark`/`exit_reason`/`net_bps`/`mfe_bps`/`mae_bps`/
`fill_source` + ASOF quote eşitliği; SQLITE_ONLY, gerçek HYBRID archive/live
sınır-geçişi, ve eksik-ASOF-kanıtı senaryolarında).

**Boundary mapping:** Oracle inclusive/inclusive (`ts_ms>=start_ms AND
ts_ms<=end_ms`); reader half-open `[start_ms, end_ms)`. Mapping: `start_ms`
DEĞİŞMEZ, yalnız `end_ms→end_ms+1`. Bu, V8/V9'daki farklı (exclusive-lower)
oracle şeklinden bilinçli olarak FARKLIdır — kör kör kopyalanmamış, ayrıca
türetilmiş ve doğrulanmıştır. Emsal (`tools/micro_edge_smoke.py`'nin
`_mark_prices_range`/`_v2` çifti) gerçek ve doğru referanslandığı doğrulandı.

**Kapsam ihlali bulundu ve düzeltildi:** İlk implementasyon (`8e8d7975`), bir
bağımsız review tarafından, yetkilendirilmiş dosya listesi DIŞINDA
`tools/range_read_inventory_reconciliation_v1_classify.py`'ye (+12 satır,
üçüncü bir `MANUAL_OVERRIDES` girdisi) tek taraflı bir değişiklik yaptığı
tespit edilerek `CORRECTIVE_IMPLEMENTATION_REQUIRED` verdiği. Git tarihçesi bu
dosyanın V1-V9'un HİÇBİRİNDE değiştirilmediğini (yalnız `f124596b`'de
yaratılıp sonra yalnız rerun edildiğini) doğruladı — bu commit bu disiplini
ilk kez kırıyordu. Operatör "geri al, sonra re-review" seçti. Düzeltme commit'i
(`15584744`) `classify.py`'yi `e4ffdc16`'ya karşı **sıfır-diff**'e geri
getirdi (bağımsız doğrulandı) ve envanter JSON'u tamamen otomatik classifier
ile yeniden üretti. Taze-context bir re-review (Kapı 4) düzeltmeyi bağımsız
olarak yeniden denetleyip **`SHADOW_EVAL_RANGE_MIGRATION_ACCEPTED`** verdi.

**Entegrasyon:** `git merge --ff-only` ile canonical'a temiz fast-forward
(`e4ffdc16..15584744`, merge commit'i YOK). Sıralama korundu: `8e8d7975`
(implementasyon) → `15584744` (kapsam düzeltmesi). Final diff yalnız 3 dosya
(`tools/s34_regime_filter_shadow_eval.py`,
`tests/test_s34_regime_filter_shadow_eval_mark_prices_reader_migration_parity.py`,
`reports/governance/storage/range_read_inventory_reconciliation_v1.json`) —
`classify.py` final diff'te YOK (net etki sıfır), foreign-owned dosyalar YOK.

**Ana ağaç doğrulaması (izole worktree'den daha güçlü — gerçek prod DB +
archive mevcut):** parity + ASOF + reader regression + production-parity
suite'leri toplam **67 passed, 0 failed, 0 skipped** (izole worktree'de 10
skip vardı, gerçek DB yokluğundan — ana ağaçta hepsi ÇALIŞTI ve geçti,
ARCHIVE_ONLY/HYBRID testleri dahil). Boundary mapping doğrudan `ami/storage/
research_reader.py` kaynağından dördüncü kez bağımsız doğrulandı (reader
`>=?/<?`, oracle `>=?/<=?`, mapping doğru).

**Envanter (doğru/gerçek):** `classification=MIGRATED_RANGE_READ`,
`manual_review_note=null`, `classification_basis=AUTOMATED`,
`reader_v2_present=true`, `v2_function_defined=true`. Sayaçlar:
`MIGRATED_RANGE_READ` 24→25, `REMAINING_MIGRATABLE` 10→9, `total_scanned=141`
(değişmedi).

**⚠️ Dürüstlük notu (ana ağaç ortam bulgusu, migrasyonun kendisiyle ilgisiz):**
Envanter scanner/classifier'ının ana ağaçta ("D:\eclipse_scalper" içinde,
oturum başından beri var olan **1109 untracked/alakasız script**) SIFIRDAN
tekrar çalıştırılması, scanner'ın `TOOLS.glob("*.py")` (git-tracked filtresi
DEĞİL, ham dosya-sistemi glob'u) kullanması nedeniyle **farklı** sayılar
üretti (total_scanned=323, REMAINING=50) — bu, izole (yalnız git-tracked
dosyaların bulunduğu) worktree ortamındaki DOĞRU/kabul-edilmiş sonuçtan farklı
bir ortam-kirliliği etkisidir, migrasyonun veya classifier'ın bir kusuru
DEĞİLDİR. Bu yanlışlıkla üretilen çıktı **HEMEN geri alındı**
(`git checkout --`), committed dosya sha256 hash'i doğrulanarak
(`26a95b55…`) önceki haline dönüldüğü teyit edildi. Sıfır-diff reprodüksiyon
zaten Kapı 4'te izole worktree'de bağımsız olarak kanıtlanmıştı (doğru,
kapsamı git-tracked dosyalarla sınırlı ortamda) — bu bulgu yalnız gelecekteki
oturumlar için bir uyarı: bu envanter script'i ana ağaçta çalıştırılmadan önce
untracked dosya kirliliği hesaba katılmalı.

**Değişmeyen:** AMI/CVD epistemik gate zayıflamadı, bilimsel çıktı değişmedi,
runtime/DB/archive/catalog/scheduler/detector/watchdog/live-execution
mutasyonu YOK, foreign-owned dosyalar dokunulmadı. **Bu kabul, başka hiçbir
MIXED_PARTIAL adayını yetkilendirmez.**

**Uygulama soy kütüğü:** `8e8d7975` (implementasyon) → `15584744` (kapsam
düzeltmesi) → bağımsız re-review `SHADOW_EVAL_RANGE_MIGRATION_ACCEPTED`.

**Verdict: `SHADOW_EVAL_RANGE_MIGRATION_ACCEPTED`.** Next:
`RETURN_TO_STORAGE_SCOPE_DECISION_FOR_REMAINING_9_CANDIDATES`.

---

## 120. RANGE-READ V11 — KALAN 9 ADAY BAĞIMSIZ YENİDEN-SINIFLANDIRMA + REVIEW ÇÜRÜTMESİ + OPERATÖR BAR ADJUDİKASYONU → ZERO ELIGIBLE (2026-07-11, Opus 4.8)

**Bağlam:** §119'un açık Next'i (`RETURN_TO_STORAGE_SCOPE_DECISION_FOR_
REMAINING_9_CANDIDATES`) ele alındı. Operatör açıkça talimat verdi: hiçbir
adaya implementasyon yetkisi verilmeden önce **kalan 9 aday temiz izole
worktree'de read-only yeniden-sınıflandırılsın + fresh-context bağımsız
review'dan geçsin**; en fazla 1 aday teknik eligible bulunursa implementation
için AYRICA açık yetki istensin. `eth_preliq_*`'nin daha önce
korumalı/CVD-adjacent işaretlendiği hatırlatıldı.

**Faz 0 — düzeltilen ilk hata:** Bu oturumun ilk (kaba) geçişi yanlışlıkla
`research_s34_eth_preliq_control.py`'yi "Tier-1 en temiz aday" önerdi. §118
tablosu ve bağımsız worktree kanıtı (`from tools.s34_shadow_paper_runner
import ...` + `FEATURE_DB = data/s34_feature_factory.db` cross-DB) bunun
`CVD_ADJACENT`/protected olduğunu doğruladı → öneri GERİ ÇEKİLDİ.

**Faz 1 — bağımsız yeniden-sınıflandırma (izole sparse worktree @ `92aeaeb1`,
yalnız tools/+ami/+governance; untracked kirlilik yok; MAX_PATH bypass):**
§118 notlarına GÜVENİLMEDEN, her diskalifiye kaynaktan yeniden doğrulandı;
production DB'ye 0 sorgu. Sonuç §118'in `NO_ELIGIBLE_CANDIDATES`'ını yeniden
üretti (9/9 diskalifiye): 1-2 predicated first-crossing forward-ASOF; 3
non-allowlisted `liquidations`; 4 `ami.constitution`+`ami.research.registry`
governance; 5 açık `ofi` composite; 6-7 shadow_paper_runner import + cross-DB
FEATURE_DB; 8 first_book_cross + through_quote SUM; 9 çok-tablolu COUNT
aggregate. (Reader satır-okuyucu; server-side SUM/COUNT/MAX yok.)

**Faz 2 — fresh-context bağımsız review (ayrı agent, salt-okunur, adversarial):
Faz 1'i ÇÜRÜTTÜ.** Review, `research_s34_btc_microtrend_sweep.py`'yi bar'ın
LİTERAL ifadesi altında **eligible** buldu: migrate edilebilir kısım
`simulate()` içinde satır 67-69 `SELECT mark_price FROM mark_prices WHERE
symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms` — `LIMIT 1` yok
(forward-ASOF değil), aggregate yok, sadece mark_price (CVD yok), governance
import yok, cross-DB yok. §118/Faz1'in MIXED_PARTIAL gerekçesi olan unbounded
`FROM liquidations` scan (:100-102) **`main()` içinde event üretimi için**;
`simulate()`'e yalnız skaler `entry_ms` akıyor → migrate edilecek range
kısmıyla **entangled DEĞİL.** Bar item 7 "non-allowlisted table access
**entangled with the range portion**" diyor → literal okumada btc_microtrend
temiz. Review ayrıca **bar tutarsızlığı** yakaladı: item 7 "entangled"
yumuşatıcısına sahip, item 6 (cross-DB) sahip değil → eth_preliq_control/
executable'ın temiz ayrışabilir `_book_series` book_ticker range read'i sırf
dosya-seviyesi cross-DB yüzünden diskalifiye; item 6 de item 7 gibi okunsaydı
onlar da tartışmalı eligible olurdu.

**Operatör adjudikasyonu — KATI DOSYA-SEVİYESİ BAR:** Faz 1 (zero) vs Faz 2
(bir eligible) çatışması bar'ın yorumuna dayandığından — ve bir dosyaya
dokunup dokunmayacağını belirlediğinden — operatör kararına götürüldü (§118/
§119 emsali: kapsam/yorum kararları operatörün). **Operatör KATI DOSYA-SEVİYESİ
okumayı seçti:** dosyada mevcut herhangi bir non-allowlisted / cross-DB /
governance erişimi, ayrışabilir olsa bile o dosyayı diskalifiye eder. Bu, item
6 ve item 7'yi **tek tip file-level** yapar (item 7'nin "entangled"
yumuşatıcısı eligibility için geçersiz; governance amacıyla file-level okunur).
Sonuç: **btc_microtrend'in literal-okuma eligible'ı yönetici bar altında
GEÇERSİZ; 9/9 INELIGIBLE.**

**Sonuç:** `V11_STRICT_FILE_LEVEL_BAR_ZERO_ELIGIBLE_REAFFIRMED`. Hiçbir
implementation, hiçbir migration başlatılmadı; `btc_microtrend`,
`eth_preliq_control`, `eth_preliq_executable` dahil hiçbir adaya dokunulmadı.
Envanter DEĞİŞMEDİ (btc_microtrend `MIXED_PARTIAL`/non-allowlisted olarak durur;
scanner main-tree'de untracked-kirlilik nedeniyle YENİDEN ÇALIŞTIRILMADI —
§119 dürüstlük notu). Kod/test/catalog/archive/DB/runtime/foreign dosya
mutasyonu YOK.

**Bar netleştirmesi (gelecek oturumlar için, operatör kararıyla):** range-read
eligibility'de non-allowlisted-table, cross-DB ve governance-coupling
diskalifiyeleri **FILE-LEVEL** uygulanır — ayrışabilirlik bir dosyayı eligible
yapmaz. "Entangled with the range portion" ifadesi bir gevşetme olarak
KULLANILMAZ.

**İnceleme zinciri kaydı:** Faz1 (classify, zero) → Faz2 (bağımsız review,
REFUTE, 1 eligible) → operatör bar adjudikasyonu (strict file-level) →
`ZERO_ELIGIBLE_REAFFIRMED`. Review'ın epistemik değeri kanıtlandı: Faz 1'in
§118 etiketini entanglement testi yapmadan kabul eden zayıflığını yakaladı;
operatör yorumu barı netleştirerek kapattı.

**Verdict: `V11_STRICT_FILE_LEVEL_BAR_ZERO_ELIGIBLE_REAFFIRMED`.** Next:
`STORAGE_RANGE_READ_MIGRATION_EXHAUSTED_UNDER_CURRENT_STRICT_BAR` — yeni
range-read migrasyon ilerlemesi ancak (a) bar'ın gevşetilmesi (operatör
kararı) veya (b) yeni/temiz bir aday ortaya çıkması ile mümkün.

---

## 121. LIQUIDATION-SILENCE DETECTOR SCHEDULING-READINESS CORRECTIVE — KABUL EDİLDİ VE ENTEGRE EDİLDİ (2026-07-11, Opus 4.8 + Sonnet 5)

**Zincir:** §114.1'in açık `DETECTOR_SCHEDULING_DECISION_PENDING` takip
maddesi üzerinden başladı. Tam kademeli inceleme zinciri (tasarım →
implementasyon → bağımsız review → correction → bağımsız re-review →
kabul) izole worktree'de (`corrective/liquidation-silence-scheduling-
readiness`, taban `5a1e1b61`) yürütüldü.

**Faz 0 — tasarım:** Salt-okunur scheduling-readiness tasarım incelemesi,
verdict `CORRECTIVE_IMPLEMENTATION_REQUIRED_BEFORE_SCHEDULING` — asıl
bulgu: `heartbeat_watchdog.py`'nin `liquidation_silence` fold'unda hiçbir
freshness (staleness) kontrolü yoktu; canlı kanıt: artefakt
`evaluated_at_utc` ~3s47dk bayat olmasına rağmen current gibi fold
ediliyordu.

**Faz 1 — corrective implementasyon:** (a) `tools/heartbeat_watchdog.py`:
`liquidation_silence` fold'u `component_fresh()` deseniyle 900s freshness
bütçesine bağlandı (900s = kabul edilen policy'nin kendi 300s
`CONTROL_STREAM_FRESH_AGE_SEC`'inden türetilen scheduler cadence'inin
3 katı). Absent artefakt → değişmeden GREEN no-op; fresh artefakt →
severity aynen geçer; stale/missing/malformed timestamp → `UNKNOWN`
(asla false-GREEN, asla dondurulmuş tarihsel RED). (b) Yeni
`tools/liquidation_silence_scheduler.py`: kabul edilmiş, DEĞİŞTİRİLMEMİŞ
`liquidation_silence_detector.run_once()`'u saran minimal persistent-loop
wrapper, default 300s cadence. (c) `start_eclipse.ps1`'e
`-EnableLiquidationSilenceScheduler` opt-in (default KAPALI, `-EnableLive`
deseniyle birebir); `status_eclipse.ps1`'e DISABLED/RUNNING/STALE_PID/
NOT_REQUESTED ayrımı; `stop_eclipse.ps1`'e scoped
`-LiquidationSilenceSchedulerOnly` durdurma. (d) 19 yeni test.

**Faz 2 — bağımsız review (1. geçiş):** Verdict
`CORRECTIVE_CHANGES_REQUIRED`. 3 MEDIUM bulgu: **F1** — 900s gerekçesi
kodda hiç kullanılmayan (dead) bir sabite ("3x established convention")
atıfta bulunuyordu, gerekçe uydurmaydı (sayının kendisi savunulabilirdi,
gerekçe değildi). **F2** — 300s cadence iki dosyada bağımsız literal
olarak tekrarlanmıştı, aralarında yalnız yorum-satırı bağı vardı, sessizce
drift edebilirdi. **F3** — `acquire_single_instance_lock()` atomik
olmayan bir read-PID→check-liveness→write-PID sırasıydı; gerçek bir
TOCTOU race'ti; mevcut testler tam da bu liveness-check'i mock'layarak
race'i gizliyordu. Ayrıca 1 LOW-MEDIUM (**F4** — PowerShell opt-in
testleri gerçek çalıştırma değil, statik kaynak-metin kontrolü) ve 2 LOW
(**F5** SIGTERM Windows'ta pratikte erişilemez, zaten dürüstçe
belgelenmiş; **F6** future-timestamp clamp davranışı önceden var olan,
bu batch'in eklemediği bir sınırlama).

**Faz 3 — correction:** F1 gerekçesi düzeltildi (dead constant atfı geri
çekildi; dürüst gerekçe: policy'nin aynı 300s değer için kullandığı
12x/24x headroom'dan kasıtlı olarak daha muhafazakâr). F2: cadence sabiti
artık `heartbeat_watchdog.py`'de `tools.liquidation_silence_scheduler`'dan
**import** ediliyor (aynı obje, `is`-identity ile test edildi; kopya
değil) — drift yapısal olarak imkansız. F3: kilit mekanizması tamamen
`msvcrt.locking()` tabanlı OS-seviyesi exclusive byte-range lock'a
değiştirildi (ayrı bir `.lock` sibling dosyasında — PID dosyasının
kendisini kilitlemenin, kilit tutulduğu sürece dosyayı aynı process
içinden bile okunamaz yaptığı, correction sırasında kendi kendine
yakalanan bir regresyon, ayrı dosyaya geçilerek düzeltildi). Eski
psutil/CIM tabanlı liveness-check makinesi dead code olarak tamamen
kaldırıldı. F4, F5, F6 kasıtlı olarak "düzeltilmedi" — dürüstçe
belgelendi (F4: gerçek PowerShell çalıştırmak test suite'in kendi
"persistent process başlatma yok" kuralını ihlal ederdi).

**Faz 4 — bağımsız re-review (2. geçiş):** Verdict
`LIQUIDATION_SILENCE_SCHEDULING_READINESS_CORRECTIVE_ACCEPTED`. F1/F2/F3
bağımsız olarak (implementasyonun kendi iddialarına güvenilmeden)
yeniden doğrulandı — policy oranları (12x/24x) sıfırdan yeniden
hesaplandı, doğru çıktı; import `is`-identity doğrulandı; **F3 için
gerçek subprocess'lerle (thread değil, `subprocess.Popen`) çapraz-process
race testi** + **gerçek `taskkill /F` hard-kill testi** yapıldı — OS
lock'un hard-kill sonrası gerçekten serbest kaldığı ve PID-content
incelemesine hiç ihtiyaç duyulmadan yeni bir acquire'ın başarılı olduğu
ampirik olarak kanıtlandı (ne implementasyonun ne de ilk correction'ın
test etmediği bir senaryo). 3 kalıntı bulgu, hepsi non-blocking, LOW/INFO,
**düzeltilmiş olarak iddia edilmiyor**:
- **RR-1** (kozmetik): birkaç docstring/yorum, repo'da mevcut olmayan bir
  path'e (`reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_PERIODIC_
  SCHEDULING_DESIGN.md`) atıfta bulunuyor — tasarım raporu yalnız geçici
  review scratchpad'inde yaşıyor, repo'ya committed değil.
- **RR-2** (miras, yeni değil): `status_eclipse.ps1`'in `Test-PidAlive`
  fallback'ı (dosyadaki tüm 12 rol için aynı, bu batch'e özgü değil)
  command-line yeniden-doğrulaması yapmıyor — dar bir pencerede PID-reuse
  status görüntüsünde yanlış RUNNING gösterebilir; gerçek single-instance
  güvenlik garantisini (Python-seviyesi OS lock) ETKİLEMİYOR.
- **RR-3** (zararsız): hard-kill sonrası `.lock` sibling dosyası diskte
  kalıntı olarak kalıyor; temizlenmiyor ama zararsız (yeni bir acquire
  bu kalıntıya rağmen başarılı — ampirik doğrulandı).

**Kapsam (tüm fazlar boyunca sabit):** yalnız 7 dosya (5 modified + 2
new): `start_eclipse.ps1`, `status_eclipse.ps1`, `stop_eclipse.ps1`,
`tests/test_liquidation_silence_detector.py`, `tools/heartbeat_watchdog.py`,
`tests/test_liquidation_silence_scheduler.py` (yeni),
`tools/liquidation_silence_scheduler.py` (yeni).
`tools/liquidation_silence_detector.py` ve `tools/liquidation_silence_
policy.py` (kabul edilmiş detector semantikleri/eşikleri) her fazda
DEĞİŞTİRİLMEDİ — sıfır-diff, her review'da bağımsız doğrulandı. 7
foreign-owned kirli dosyaya (primary worktree'deki uncommitted içerik)
hiçbir fazda dokunulmadı.

**⚠️ ÖNEMLİ DÜRÜSTLÜK NOTU — bu kayıt bir aktivasyon kaydı DEĞİLDİR:**
scheduler **default KAPALI** kalıyor. Bu batch boyunca hiçbir noktada
scheduler başlatılmadı, hiçbir Scheduled Task oluşturulmadı, hiçbir canlı
runtime prosesi (collector/watchdog/executor) başlatılmadı veya yeniden
başlatılmadı, `-EnableLiquidationSilenceScheduler` hiçbir gerçek
`start_eclipse.ps1` çağrısında kullanılmadı. Sürekli izleme ŞU AN
ÇALIŞMIYOR. Aktivasyon, bu kaydın kapsamı dışında, ayrı ve açık bir
operatör kararı gerektirir.

**Entegrasyon:** implementasyon commit'i (`70d8f70e`, "feat(ops): harden
liquidation-silence scheduling readiness") + bu governance kaydı,
canonical `codex/data-layer-fallback-cleanup`'a temiz fast-forward ile
entegre edildi (ayrı, temiz bir integration worktree'den — foreign-dirty
primary worktree'den DEĞİL) ve push edildi. Detay: bkz. entegrasyon
sonrası bu bölümün altına eklenecek push doğrulaması (varsa) veya ayrı
governance kaydı.

**Verdict: `LIQUIDATION_SILENCE_SCHEDULING_READINESS_ACCEPTED_AND_
INTEGRATED`.** Next: `OPERATOR_ADJUDICATION_FOR_LIQUIDATION_SILENCE_
SCHEDULER_ACTIVATION` — periyodik aktivasyon (gerçek `-EnableLiquidation
SilenceScheduler` çağrısı ile) ayrı, açık bir operatör kararı ve kendi
kademeli inceleme zincirini gerektirir; bu kayıt onu YETKİLENDİRMEZ.

---

## 122. GATE 1 — LIQUIDATION-SILENCE SCHEDULER CANARY — KABUL EDİLDİ (2026-07-11, Sonnet 5)

**Zincir:** §121'in `OPERATOR_ADJUDICATION_FOR_LIQUIDATION_SILENCE_
SCHEDULER_ACTIVATION` Next'ine yanıt olarak operatör, `-EnableLiquidation
SilenceScheduler` ile kontrollü, süreli bir Gate 1 canary çalıştırmasını
açıkça yetkilendirdi. Canary çalıştırıldı, sonuçları BAĞIMSIZ bir
operasyonel review agent'ı tarafından (bu implementasyonu üreten oturumdan
ayrı, salt-okunur, kanıt-önce) denetlendi — kademeli inceleme zincirinin
epistemik gereği ([[feedback_gated_independent_review_chain]]) böylece
karşılandı.

**Canary parametreleri:** scheduler PID `15640`, başlangıç ~
`2026-07-11T17:40:51Z`; istenen pencere 30–45 dakika; gözlenen süre
~34 dakika, pencere içinde. Canary sonunda scheduler, scoped
`stop_eclipse.ps1 -LiquidationSilenceSchedulerOnly` yolu ile durduruldu
(kesin scoped-stop UTC zaman damgası hiçbir yerde doğrudan loglanmadı —
bu bir iddia değil, bilinen bir kanıt sınırı).

**Cycle sonuçları:** `7/7` ardışık `SUCCESS` (gerekli asgari 5'in üzerinde).
Yedi cycle'ın tamamı `GREEN`/`HEALTHY`, `reason_codes` boş. İstisna yok,
timeout yok, malformed cycle yok, overlap yok, tight-loop yok, sessiz
scheduler ölümü yok, scheduler restart'ı yok, `STALE_ARTIFACT` tekrarı yok.

**Cadence düzeltmesi (kanonik, önceki iki hatalı formülasyonun yerine):**
altı ardışık-cycle aralığının tamamı ~300.12 saniyeydi ve en yakın tam
saniyeye yuvarlandığında hepsi 300 saniyeye yuvarlanıyor; minimum 300.114s,
maksimum 300.131s, ortalama ~300.122s. Önceki cadence-sayım ifadesi
("300s×6, 301s×1" ve ardından "five rounded to 300 seconds and one
rounded to 301 seconds") 7 cycle için 7 (aslında 6) aralık sayma
hatasından kaynaklanan sadece bir raporlama/aritmetik hatasıydı — gerçek
bir scheduler zamanlama kusurunu TEMSİL ETMİYORDU.

**Artefakt tazeliği:** `liquidation_silence.json` ve `overall.json`
scheduler cycle'larıyla birlikte ilerledi; sağlık-artefaktı hash'leri
gözlenen cycle-count ilerlemesiyle lockstep değişti; gözlenen çıktı
donmuş/bayat-sonuç yeniden kullanımı değil, taze yeniden-değerlendirme
gösterdi; `evaluated_at_utc` monoton ilerledi; `STALE_ARTIFACT` canary
boyunca tekrar etmedi. (Not: bağımsız review yalnız mevcut ara-anlık
(intermediate) izleme kanıtının kapsadığı hash geçişlerini doğruladı —
her tarihsel hash'in ayrı ayrı bağımsız kurtarıldığı iddia edilmiyor.)

**Süreç/runtime izolasyonu:** scheduler süreç sayısı aktif pencere
boyunca tam olarak bir kaldı; scheduler PID'i sürekli `15640`, kimliği ve
StartTime'ı PID-metadata'dan doğrulandı; hiçbir çift/yedek scheduler
instance'ı bulunmadı; hiçbir scheduler restart'ı olmadı. `heartbeat_
watchdog` sürekli PID `9740` kaldı, StartTime değişmedi, çıktısı
ilerlemeye devam etti. Diğer on bir yönetilen runtime rolü (collector_
supervisor, bookticker_collector, oi_spot_poller, s34_live_chart,
microstructure_collector, s34_shadow_paper_runner, s34_realtime_shadow_
runner, s34_v_engine_v02_shadow_mirror, event_diary, orderflow_chart,
s34_replay) mevcut izleme kanıtı boyunca değişmeyen kimliklerle canlı
kaldı. Her iki live executor kapalı (OFF) kaldı. Scheduler stderr boş
kaldı. Scheduler CPU/memory/log büyümesi sınırlı (bounded) kaldı. Scoped
stop yalnız scheduler'ı etkiledi.

**Monitor false-positive bulgusu:** canary sırasında salt-okunur bir
izleme scripti bir false-positive süreç-sayısı abort'u üretti. Kabul
edilen açıklama: PowerShell'in tekil-nesne `.Count` davranışı tek bir
eşleşen süreç için `null` döndürdü; script `@()` array-wrap ile
düzeltildi; o ana ait kanıtlar (PID 15640 canlı ve tekil, StartTime
değişmemiş, stderr boş, log/artefakt ilerlemesi kesintisiz) gerçek bir
scheduler kesintisi olmadığını gösterdi; false alarm nedeniyle hiçbir
scheduler aksiyonu alınmadı. Kanıt sınırı: monitor dosyası yerinde
düzenlendiği için tam düzeltme-öncesi (pre-fix) kaynak satırı kurtarılamadı
— bu satırın bağımsız olarak kurtarıldığı iddia edilmiyor.

**Kabul edilen, bloklayıcı olmayan bulgular** (Gate 1 kabulünü
engellemiyor; düzeltici implementasyon bu kaydın kapsamı dışında, ayrı
yetki gerektirir):
- **MEDIUM** — monitor harness scheduler canlılığını öncelikle PID ile
  kontrol etti, her snapshot'ta StartTime'ı yeniden okumadı; teorik bir
  PID-reuse kör noktası bırakıyor.
- **LOW** — kesin scoped-stop olay zaman damgası doğrudan loglanmadı.
- **LOW** — tam düzeltme-öncesi monitor-script satırı, dosya yerinde
  düzenlendiği için kurtarılamadı.
- **LOW** — orijinal cadence özeti, kanonik Gate 1 kaydında düzeltilen,
  önemsiz (non-substantive) bir aritmetik ifade hatası içeriyordu.

CRITICAL bulgu YOK. HIGH bulgu YOK.

**Post-stop / default-OFF durumu:** scheduler süreç sayısı = 0; scheduler
PID dosyası yok; bilinen sıfır-byte `.pid.lock` kalıntısı duruyor (zararsız
ve bloklayıcı olmayan olarak sınıflandırıldı, kaldırılmadı); watchdog PID
9740 sağlıklı; diğer on bir runtime rolü canlı; her iki live-executor
marker'ı `0`; scheduler aktivasyonu açık `-EnableLiquidationSilence
Scheduler` anahtarını gerektiriyor, bu anahtar olmadan scheduler default
KAPALI kalıyor; hiçbir Scheduled Task yok.

**Repository/reconciliation bağlamı (özet):** Gate 1 kabul kaydından önce,
branch kontrollü, saf bir fast-forward ile uzlaştırıldı — kanonik taban
`5a1e1b61...`'den `2999f228...`'e ilerledi, uzak scheduling-readiness
commit'leri (`70d8f70e`+`2999f228`) merge-commit'siz entegre edildi.
Çakışan lokal taslaklar (`start_eclipse.ps1`, `status_eclipse.ps1`,
`stop_eclipse.ps1`, `tools/heartbeat_watchdog.py`) ve çakışan untracked
`tools/liquidation_silence_scheduler.py`, kanonik uzak versiyonlar
benimsenmeden ÖNCE repo dışında byte-seviyesinde korundu. Reconciliation
sırasında hiçbir runtime restart'ı olmadı; ilgisiz beş dirty tracked path
dokunulmadan kaldı. Tam adli kanıt iki harici arşivde saklanıyor (bu
kayda dahil edilmedi, arşivler bloat önlemek için burada tekrarlanmıyor):
`D:\eclipse_scalper_reconciliation_quarantine\gate1_pre_ff_
20260711T185433Z\` ve `D:\eclipse_scalper_reconciliation_quarantine\
gate1_collision_clear_20260711T190113Z\`.

**Yetkilendirme sınırı:** Gate 1 KABUL EDİLDİ. Scheduler durdurulmuş
durumda kalıyor. Scheduler default KAPALI kalıyor. Bu kabul: Gate 2'yi
YETKİLENDİRMEZ; kalıcı scheduler etkinleştirmesini YETKİLENDİRMEZ;
Scheduled Task oluşturmayı YETKİLENDİRMEZ; runtime restart'ı
YETKİLENDİRMEZ; monitor-harness düzeltici implementasyonunu
YETKİLENDİRMEZ. Sonraki her runtime veya düzeltici aksiyon ayrı, açık bir
operatör yetkisi gerektirir.

**Verdict: `GATE_1_ACCEPTED`.** Kanonik post-recording durum:
`GATE_1_ACCEPTED_SCHEDULER_STOPPED_DEFAULT_OFF_AWAITING_SEPARATE_GATE_2_
AUTHORIZATION`. Next: Gate 2 (kalıcı/uzun-süreli aktivasyon) yalnız ayrı,
açık bir operatör kararı ve kendi kademeli inceleme zinciriyle
başlatılabilir; bu kayıt onu başlatmaz.

## 123. GATE 1 MONITOR-HARNESS — LIQUIDATION-SILENCE CANARY MONITOR SCRIPT'İ — 4 TUR BAĞIMSIZ İNCELEME SONRASI KABUL EDİLDİ (2026-07-12, Sonnet 5)

**Zincir:** §122'de Gate 1 canary kabulü sırasında flagged edilen MEDIUM
bulgu (monitor harness'ın scheduler canlılığını yalnız PID ile kontrol
etmesi, StartTime'ı her snapshot'ta yeniden okumaması — teorik PID-reuse
kör noktası) ve o canary'de kullanılan salt-okunur izleme script'inin
kalıcı, yeniden kullanılabilir, tam test edilmiş bir modüle
(`tools/liquidation_silence_canary_monitor.py` +
`tests/test_liquidation_silence_canary_monitor.py`) taşınması kararı
doğrultusunda, [[feedback_gated_independent_review_chain]] disiplinine
tam uyumlu implementation→review→correction→re-review zinciri 4 tur
boyunca yürütüldü. Her fazın uygulayıcısı ve bağımsız reviewer'ı ayrı
agent/oturum geçişleriydi (review'lar her seferinde implementasyonu
üreten oturumdan ayrı, salt-okunur, taze bir agent tarafından yürütüldü);
hiçbir faz kendi kendini onaylamadı.

**Tur özeti:**
- **Round 1/2** (bu oturumun kapsamı dışında, önceki oturumlarda
  tamamlandı): F-01..F-10 (StartTime string karşılaştırması, identity
  truthiness-only, unanchored substring matching, unvalidated StopEvent,
  dishonest git-commit provenance, tautological hash testi, cadence
  monotonicity eksikliği, whole-second timestamp kullanımı, CLI
  continuity her-zaman-null, `sample_artifact()` untested/raised-on-
  directory) kapatıldı.
- **Round 3** (implementation→review→correction→re-review): F-BASELINE-01,
  F-READFAIL-01, F-MIXEDPREC-01, F-ROUND-01 kapatıldı; F-STOP-01'in
  çekirdek riski (StopEvent doğrudan construction bypass'ı) kapatıldı,
  ancak bağımsız re-review bir MEDIUM (StopEvent doğrulamasının yalnız
  `STOP_VERIFIED_ABSENT` outcome'ı için uygulanması — diğer outcome'larda
  çelişkili kanıt hâlâ inşa edilebiliyordu) ve bir LOW
  (`--scheduler-log-path` bir dizine işaret ettiğinde unhandled
  `PermissionError`) bulgusuyla `GATE_1_MONITOR_HARNESS_ROUND_3_
  CORRECTIVE_REQUIRED` verdict'i verdi.
- **Round 4** (implementation→independent re-review): F-STOP-COMPAT-01
  (tam outcome/verification uyumluluk matrisi — `_STOP_OUTCOME_RULES`,
  `StopEvent.__post_init__` içinde 6 outcome'ın tamamı için enforce
  edildi) ve F-CYCLELOG-01 (`load_cycle_log()` tipli, fail-closed
  cycle-log loader'ı; `parse_cycle_log()` geriye-uyumlu ince bir
  wrapper'a indirgendi; `take_snapshot()` çıktısına açık `cycle_log`
  alanı eklendi) kapatıldı. Bağımsız re-review
  `GATE_1_MONITOR_HARNESS_ROUND_4_ACCEPTED` verdict'ini verdi.

**Eklenen test kapsamı:** Round 4'te **52 yeni test** eklendi — **31**
StopEvent uyumluluk-matrisi testi (6 outcome'ın her biri için pozitif
construction + 21 çelişki-reddi senaryosu + builder/direct parity +
mutation probe'lar) ve **21** cycle-log fail-closed testi (10 gerekli
senaryo + typed-result/legacy-wrapper parity + mutation probe'lar).
Toplam odaklanmış (focused) test sayısı 163 → 215'e çıktı. (Not:
implementasyon raporu ilk aşamada "32 StopEvent testi" olarak yanlış
saydı — bağımsız review 31 olduğunu doğruladı; 31+21=52, 163→215
artışıyla tam uyuşuyor — bu bir raporlama düzeltmesidir, kod kusuru
değildir.)

**Bağımsız regresyon kanıtı** (Round 4 re-review tarafından bizzat
çalıştırıldı, implementasyon raporundan alınmadı):
- odaklı monitor testleri: **215 passed**
- monitor + scheduler: **226 passed**
- heartbeat watchdog + detector: **73 passed**
- liquidation-silence policy + native-WS policy: **69 passed**
- `python -B -m py_compile` (her iki dosya): **temiz**

**Kabul edilen, bloklayıcı olmayan bulgular** (Round 4 kabulünü
engellemiyor; ayrı yetki olmadan bu kayıt kapsamında düzeltilmedi):
- **LOW (F-1)** — `StopEvent` mutable bir dataclass (`frozen=True`
  değil); construction sonrası doğrudan attribute mutation
  `__post_init__`'i yeniden tetiklemiyor, teorik olarak çelişkili bir
  kayıt üretebilir. Bağımsız review bunun mevcut kodda **latent, exploit
  edilmemiş** olduğunu doğruladı (hiçbir çağıran construction-sonrası
  mutation yapmıyor; model henüz canlı stop-orchestration'a bağlı değil).
  Önerilen düzeltme: bu model canlı stop-control'e entegre edilmeden
  ÖNCE `frozen=True` uygulanmalı — bu geçişte UYGULANMADI.
- **INFORMATIONAL (F-2)** — `parse_cycle_log()` içinde zararsız,
  ulaşılamaz (dead) bir `return` satırı.
- **LOW (F-3)** — "CLI legacy `parse_cycle_log()` wrapper'ına geri döner"
  regresyonu için özel bir mutation-probe testi yok; bağımsız review
  gerçek riskin `AttributeError` ile anında ve gürültülü şekilde ortaya
  çıkacağı için yapısal olarak düşük olduğunu değerlendirdi.

CRITICAL bulgu YOK. HIGH bulgu YOK. Çözülmemiş MEDIUM bulgu YOK.

**Repository/runtime sınırları** (implementasyon ve her iki bağımsız
review boyunca korundu, bu governance-only kayıt geçişinde de yeniden
doğrulandı): her iki düzeltici dosya (`tools/liquidation_silence_canary_
monitor.py` — 1966 satır, `tests/test_liquidation_silence_canary_
monitor.py` — 2701 satır) untracked kaldı; beş korumalı dirty tracked
path (`.claude/settings.local.json`, `runtime/dashboard_backend.json`
[silinmiş], `tests/test_native_ws_health_policy.py`, `tools/native_ws_
health_policy.py`, `tools/s34_cascade_navigation_dashboard.py`)
byte-seviyesinde değişmedi; staging boş kaldı; hiçbir commit/push
olmadı. Scheduler süreç sayısı = 0 kaldı; `.pid.lock` sıfır-byte kaldı;
watchdog PID 9740 ve CreationDate değişmedi; diğer on bir runtime rolü
canlı kaldı; her iki live-executor marker'ı `0` kaldı; hiçbir Scheduled
Task oluşturulmadı; hiçbir runtime script (`start_eclipse.ps1`/
`status_eclipse.ps1`/`stop_eclipse.ps1`) çağrılmadı; iki harici
reconciliation arşivi (`gate1_pre_ff_20260711T185433Z\`,
`gate1_collision_clear_20260711T190113Z\`) dokunulmadan kaldı.

**Yetkilendirme sınırı:** Bu kayıt yalnız GOVERNANCE'tır. Şu ana kadar
staging/commit/push YAPILMADI (kod hâlâ untracked). Gate 2 aktivasyonunu
YETKİLENDİRMEZ; kalıcı scheduler etkinleştirmesini YETKİLENDİRMEZ;
Scheduled Task oluşturmayı YETKİLENDİRMEZ; runtime restart'ı
YETKİLENDİRMEZ; live-executor aktivasyonunu YETKİLENDİRMEZ; F-1'in
`frozen=True` düzeltmesini bu geçişte UYGULAMAZ. Sonraki her aksiyon
(commit dahil) ayrı, açık bir operatör yetkisi gerektirir.

**Verdict: `GATE_1_MONITOR_HARNESS_ROUND_4_ACCEPTED`.** Kanonik
post-recording durum: `GATE_1_MONITOR_HARNESS_ACCEPTANCE_RECORDED_
AWAITING_COMMIT_AUTHORIZATION`. Next: kod hâlâ untracked; staging+commit
(F-1'in `frozen=True` düzeltmesi dahil veya hariç, operatör tercihi) ayrı,
açık bir operatör kararını bekliyor.

## 124. STOPEVENT IMMUTABILITY HARDENING (F-1) — BAĞIMSIZ İNCELEME SONRASI KABUL EDİLDİ (2026-07-12, Sonnet 5)

**Zincir:** §123'te kaydedilen §122 kabulünün LOW F-1 bulgusuna
("`StopEvent` mutable bir dataclass; construction sonrası doğrudan
attribute mutation `__post_init__`'i yeniden tetiklemiyor, teorik olarak
çelişkili bir kayıt üretebilir") yanıt olarak, dar kapsamlı bir düzeltici
micro-batch yürütüldü ve ardından, [[feedback_gated_independent_review_
chain]] disiplinine tam uyumlu şekilde, implementasyonu üreten oturumdan
tamamen ayrı, salt-okunur, kanıt-önce bir agent tarafından bağımsız
re-review yapıldı. Verdict: `STOP_EVENT_IMMUTABILITY_HARDENING_ACCEPTED`.

**Amaç ve kapanış:** Kabul edilmiş LOW F-1 kapatıldı. `StopEvent`
(`tools/liquidation_silence_canary_monitor.py`) artık
`@dataclass(frozen=True)` olarak tanımlı. Construction-sonrası field
assignment ve field deletion artık `dataclasses.FrozenInstanceError` ile
engelleniyor. Orijinal invariant-bypass mutation senaryosu (geçerli bir
`STOP_VERIFIED_ABSENT` event'i construct ettikten sonra
`event.outcome = STOP_FAILED` gibi doğrudan bir atama ile çelişkili bir
kayıt üretmek) artık mümkün değil.

**Bağımsız kaynak-inceleme kanıtı:**
- `StopEvent`'in modül içinde tam olarak tek bir kanonik tanımı var.
- `__post_init__` hiçbir field mutation'ı yapmıyor (yalnız okuma + raise)
  — frozen dataclass semantiğiyle yapısal olarak uyumlu.
- İlgili implementasyon ve test dosyalarının tamamında `object.
  __setattr__`, construction-sonrası mutation, mutation helper fonksiyonu
  veya `StopEvent`'in mutable olduğunu varsayan bir kod bulunmadı.
- Yalnız `StopEvent` hardened edildi; modüldeki diğer 13 dataclass
  değişmeden kaldı.
- `parse_cycle_log()` içindeki bilinen dead `return` satırı ve eksik
  legacy-wrapper mutation probe'u, bu micro-batch'in kasıtlı olarak
  kapsamı dışında kaldı (dokunulmadı).

**Bağımsız davranışsal kanıt:** Bağımsız reviewer'ın kendi yazdığı,
repo dışı bir throwaway script ile çalıştırdığı **34/34 probe PASS**:
6 geçerli outcome'ın tamamı construct edildi; 21 çelişkili kombinasyonun
tamamı `ValueError` fırlattı; builder, direct, payload ve `dataclasses.
replace()` yolları geçerli kaldı; uyumlu `replace()` çağrıları başarılı
oldu, çelişkili `replace()` çağrıları fail-closed reddedildi (yani
`replace()` her zaman `__init__`/`__post_init__`'i yeniden çalıştırıyor,
frozen olsa da bypass değil); 7 `StopEvent` alanının tamamı için hem
assignment hem deletion `FrozenInstanceError` fırlattı; başarısız
mutation denemeleri sonrası nesne tamamen değişmeden kaldı (asdict
snapshot'ları deneme öncesi/sonrası byte-bybyte özdeşti).

**Bağımsız regresyon kanıtı** (implementasyon raporundan değil, bağımsız
reviewer'ın kendi çalıştırdığı komutlardan):
- odaklı monitor testleri: **226 passed**
- monitor + scheduler: **237 passed**
- heartbeat watchdog + detector: **73 passed**
- liquidation-silence policy + native-WS policy: **69 passed**
- `python -B -m py_compile` (her iki dosya): **temiz**

**Kabul edilen, bloklayıcı olmayan bulgu:**
- **INFORMATIONAL** — `frozen=True`, Python'ın `eq=True`/`frozen=True`
  varsayılan davranışı gereği otomatik bir `__hash__` üretiyor; ancak
  `StopEvent`, mutable/unhashable bir `target_pids: List[int]` alanı
  taşıdığı için `hash(event)` çağrısı hâlâ `TypeError` fırlatıyor (mesaj
  metni `'StopEvent'` yerine `'list'` oldu — davranışsal fark yalnız
  hata mesajında, işlevsel bir regresyon değil). Hiçbir production veya
  test kodu `StopEvent`'i bir `set` üyesi veya `dict` key'i olarak
  kullanmıyor — bu kabul için düzeltici aksiyon gerekmiyor.

CRITICAL bulgu YOK. HIGH bulgu YOK. Çözülmemiş MEDIUM bulgu YOK.

**Repository/runtime sınırları** (implementasyon ve bağımsız review
boyunca korundu, bu governance-only kayıt geçişinde de yeniden
doğrulandı): `tools/liquidation_silence_canary_monitor.py` (1976 satır,
SHA-256 `44ac9bd5...`) ve `tests/test_liquidation_silence_canary_
monitor.py` (2932 satır, 226 test fonksiyonu, SHA-256 `350d6620...`)
untracked kaldı; beş korumalı dirty tracked path byte-seviyesinde
değişmedi; staging boş kaldı; hiçbir commit/push olmadı. Scheduler süreç
sayısı = 0 kaldı; `.pid.lock` sıfır-byte kaldı; watchdog PID 9740 ve
CreationDate değişmedi; diğer on bir runtime rolü canlı kaldı; her iki
live-executor marker'ı `0` kaldı; hiçbir Scheduled Task oluşturulmadı;
hiçbir runtime script çağrılmadı.

**Yetkilendirme sınırı:** Bu kayıt yalnız GOVERNANCE'tır. Ne implementasyon
ne de test dosyası bu geçişte değiştirildi. Şu ana kadar staging/commit/
push YAPILMADI (kod hâlâ untracked). Gate 2 aktivasyonunu YETKİLENDİRMEZ;
kalıcı scheduler etkinleştirmesini YETKİLENDİRMEZ; Scheduled Task
oluşturmayı YETKİLENDİRMEZ; runtime restart'ı YETKİLENDİRMEZ;
live-executor aktivasyonunu YETKİLENDİRMEZ; ek hardening (`target_pids`
tipini veya hashing davranışını değiştirmek gibi) UYGULAMAZ. Sonraki her
aksiyon (commit dahil) ayrı, açık bir operatör yetkisi gerektirir.

**Verdict: `STOP_EVENT_IMMUTABILITY_HARDENING_ACCEPTED`.** Kanonik
post-recording durum: `STOP_EVENT_IMMUTABILITY_HARDENING_ACCEPTANCE_
RECORDED_AWAITING_COMMIT_AUTHORIZATION`. Next: kod hâlâ untracked;
staging+commit ayrı, açık bir operatör kararını bekliyor.
