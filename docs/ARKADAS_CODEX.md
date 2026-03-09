# ARKADAS CODEX — Arkadasin Claude'una Verilecek Talimat

> Bu dosyayi arkadasin Claude'una ver. O tum projeyi tarayip `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` olusturacak.

---

## TALIMAT

Sen Eclipse Scalper botunun **Kisi 1 (Research/Data)** tarafinin Claude'usun.
Gorev: Tum projeyi tara ve asagidaki formatta bir rapor olustur.

Raporu `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` dosyasina yaz.

---

## PROJE HAKKINDA

- **Repo:** Eclipse Scalper — Binance Futures USDT-M Perpetual scalping bot
- **Tech:** Python asyncio + ccxt
- **2 kisilik ekip:**
  - **Kisi 1 (sen/arkadasim):** data/, features/, strategies/, tools/, tests/ (Research/Data)
  - **Kisi 2 (ben):** execution/, risk/, bot/, exchanges/, notifications/, dashboard/, monitoring/ (Runtime/Ops)
- **Paylasilan:** config/, docs/, README.md, .github/, scripts/
- **Kural:** Birbirimizin klasorlerine dokunmuyoruz (koordinasyon olmadan)

---

## RAPOR FORMATI

Asagidaki basliklar altinda rapor olustur:

### 1. GENEL DURUM
- Hangi branch'tesin?
- Son commit hash ve mesaji
- Toplam dosya sayisi (sadece .py)
- Toplam test sayisi (pytest --co -q ile say)

### 2. KISI 1 ALANLARI — DETAYLI TARAMA
Her dizin icin:
- `strategies/` — Her .py dosyasini listele (dosya adi, satir sayisi, ne yapiyor 1 cumle)
- `data/` — Her .py dosyasini listele (dosya adi, satir sayisi, ne yapiyor)
- `tools/` — Sadece ana tool'lari listele (20+ dosya varsa en onemli 15'i)
- `tests/` — Kac test dosyasi, kac test fonksiyonu, kac PASS/FAIL

### 3. KISI 2 ALANLARI — OZET TARAMA
Sadece su dizinlerdeki dosya listesi ve satir sayilari:
- `execution/` — Her .py dosyasi (ad + satir)
- `bot/` — Her .py dosyasi
- `exchanges/` — Her .py dosyasi
- `notifications/` — Her .py dosyasi
- `dashboard/` — Her .py dosyasi
- `monitoring/` — Her .py dosyasi
- `risk/` — Her .py dosyasi

### 4. PAYLASILAN DOSYALAR
- `config/` icerigini listele
- `utils/` icerigini listele
- `core/` icerigini listele
- `brain/` icerigini listele

### 5. EXECUTION DOSYALARI KARSILASTIRMA
Bu cok onemli. Kisi 1'in branch'inde (`feat/execution-hardening-and-strategies`) execution/ altinda su dosyalar var (veya yeni eklendi):
- circuit_breaker.py
- event_journal.py
- flatten_intent.py
- intent_ledger_persistence.py
- order_verifier.py
- position_lock.py
- rate_limiter.py
- health_monitor.py
- metrics_collector.py
- system_status.py
- protection_manager.py
- state_machine.py

Her biri icin:
- Dosya var mi? Satir sayisi?
- Main branch'teki ayni dosya ile fark var mi?
- Dosyanin ana gorevi ne?

### 6. TODO / FIXME / HACK
Tum projede `TODO`, `FIXME`, `HACK`, `XXX` yorumlarini tara.
Her biri icin: dosya:satir — yorum metni

### 7. IMPORT CAKISMALARI
Asagidaki modullerin birden fazla implementasyonu var mi kontrol et:
- `health_monitor` vs `guardian` — ikisi de watchdog mi?
- `metrics_collector` vs `telemetry` — ikisi de metrik mi topluyor?
- `system_status` vs `status_snapshot` — ikisi de durum mu raporluyor?
- `protection_manager` vs `kill_switch` — ikisi de koruma mi?
- `state_machine` — iki versiyon var mi?

### 8. GIT DURUMU
- `git status` ciktisi
- Commit edilmemis degisiklik var mi?
- Remote ile senkron mu? (`git log origin/HEAD..HEAD`)

### 9. CIKARIM ve ONERI
- Kisi 2 (ben) ile senkron olmak icin ne yapmamiz lazim?
- Hangi dosyalar cakisiyor?
- Merge stratejisi onerisi

---

## ONEMLI NOTLAR

1. Raporu `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` olarak kaydet
2. Turkce yaz ama dosya adlari ve kod referanslari Ingilizce kalsin
3. Satir sayilarini `wc -l` ile al
4. Test sayilarini `python -m pytest --co -q 2>&1 | tail -5` ile al
5. Guardian-safe prensibi: Hicbir dosyayi DEGISTIRME, sadece OKU ve RAPORLA
6. `docs/CLAUDE.md` dosyasini oku ve anla — bu projenin operasyonel doktrini
7. Bu rapor Kisi 2'nin Claude'u tarafindan analiz edilecek — detayli ve dogru ol
