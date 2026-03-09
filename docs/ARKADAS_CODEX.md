# ARKADAS CODEX — Bu dosyayi CLAUDE.md olarak kullan

> Bu dosyayi proje klasorune CLAUDE.md olarak koy veya codex'ine yapistir.
> Claude otomatik olarak projeyi tarayip `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` olusturacak.

---

## SEN KIMSIN

Sen Eclipse Scalper botunun **Kisi 1 (Research/Data)** tarafinin Claude'usun.

## PROJE HAKKINDA

- **Repo:** Eclipse Scalper — Binance Futures USDT-M Perpetual scalping bot
- **Tech:** Python asyncio + ccxt
- **GitHub:** https://github.com/phoenixsenses/eclipse_scalper.git
- **2 kisilik ekip:**
  - **Kisi 1 (sen):** data/, features/, strategies/, tools/, tests/ (Research/Data)
  - **Kisi 2 (arkadasin):** execution/, risk/, bot/, exchanges/, notifications/, dashboard/, monitoring/ (Runtime/Ops)
- **Paylasilan:** config/, docs/, README.md, .github/, scripts/, utils/, core/, brain/
- **Kural:** Birbirimizin klasorlerine dokunmuyoruz (koordinasyon olmadan)

## OPERASYONEL DOKTRIN

Bu proje `docs/CLAUDE.md` dosyasinda tanimlanmis kapsamli bir operasyonel doktrine sahiptir.
Ilk is olarak `docs/CLAUDE.md` dosyasini oku ve anla. Oradaki kurallar bu talimattan onceliklidir.

Ozetle:
- Intent-driven execution model (order router uzerinden)
- Eventually consistent reconciliation (reconcile.py tek gercek kaynagi)
- Guardian-safe contract (hicbir fonksiyon exception firlatmaz)
- Kill-switch -> Circuit breaker -> Risk manager -> Entry gates -> Router -> Exchange hiyerarsisi
- Deterministic research (ayni input + seed = ayni output)
- `docs/INVARIANTS.md` deki tum invariant'lar korunmali

## GOREV: PROJE TARAMASI

Asagidaki adimlari sirayla yap ve sonucu `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` dosyasina yaz.
Hicbir dosyayi DEGISTIRME, sadece OKU ve RAPORLA (bu dosya haric).

### ADIM 1: GIT DURUMU
Su komutlari calistir ve ciktilarini rapora yaz:
```bash
git branch --show-current
git log --oneline -5
git status
git log origin/main..HEAD --oneline
git diff --stat origin/main..HEAD
```

### ADIM 2: KISI 1 ALANLARI — DETAYLI TARAMA

**strategies/** — Her .py dosyasi icin:
- Dosya adi, satir sayisi (`wc -l`), tek cumle aciklama

**data/** — Her .py dosyasi icin:
- Dosya adi, satir sayisi, tek cumle aciklama
- Alt klasorleri de tara (data/features/, data/labels/, data/derived/, data/live/)

**tools/** — En onemli 20 dosya:
- Dosya adi, satir sayisi, tek cumle aciklama
- 20'den fazla varsa geri kalanlari sadece say

**tests/** — Ozet:
- Toplam test dosya sayisi
- `python -m pytest --co -q 2>&1 | tail -5` ciktisi (test sayisi)
- Bilinen failure'lar varsa listele

### ADIM 3: KISI 2 ALANLARI — OZET TARAMA

Su dizinlerdeki her .py dosyasinin adini ve satir sayisini listele:
- execution/
- bot/
- exchanges/
- notifications/
- dashboard/ (ve alt klasorleri)
- monitoring/
- risk/

### ADIM 4: PAYLASILAN DOSYALAR

Su dizinlerin icerigini listele (dosya adi + satir sayisi):
- config/
- utils/
- core/
- brain/

### ADIM 5: EXECUTION DOSYALARI KARSILASTIRMA

Bu cok onemli — Kisi 2'nin branch'i (main) ile senin branch'in arasindaki farklari bulmamiz lazim.

Asagidaki 12 dosya Kisi 1 tarafindan eklenmis olabilir. Her biri icin:
- Dosya var mi? Satir sayisi?
- `git log --oneline -3 -- <dosya>` ile son 3 commit
- Dosyanin ana gorevi ne? (ilk 20 satiri oku)

Dosyalar:
1. execution/circuit_breaker.py
2. execution/event_journal.py
3. execution/flatten_intent.py
4. execution/intent_ledger_persistence.py
5. execution/order_verifier.py
6. execution/position_lock.py
7. execution/rate_limiter.py
8. execution/health_monitor.py
9. execution/metrics_collector.py
10. execution/system_status.py
11. execution/protection_manager.py
12. execution/state_machine.py

Ek olarak su 3 dosya Kisi 2 tarafindan eklendi — sende var mi kontrol et:
13. execution/preflight.py
14. execution/env_sanity.py
15. execution/shared_locks.py

### ADIM 6: TODO / FIXME / HACK TARAMASI

Tum projede su pattern'leri ara:
```bash
grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.py" | head -50
```
Her birini rapora yaz: `dosya:satir — yorum metni`

### ADIM 7: IMPORT CAKISMALARI

Asagidaki modullerin birden fazla implementasyonu var mi kontrol et:
- `health_monitor` vs `guardian` — ikisi de watchdog mi?
- `metrics_collector` vs telemetry modulleri — ikisi de metrik mi topluyor?
- `system_status` vs `status_snapshot` — ikisi de durum mu raporluyor?
- `protection_manager` vs `kill_switch` — ikisi de koruma mi?
- `state_machine` — birden fazla versiyon var mi?

Her biri icin: hangi dosyada, kac satirlik, ne yapiyor, cakisma var mi?

### ADIM 8: STRATEGIES DETAY

strategies/ altindaki her dosyanin:
- Import etigi modulleri listele
- Disaridan cagrilan public fonksiyonlari listele
- Hangi indicator'lari kullandigini belirt
- Test dosyasi var mi? (tests/ altinda ara)

### ADIM 9: CIKARIM ve ONERI

Tarama sonucunda su sorulari cevapla:
1. Kisi 2 (main branch) ile senkron muyuz? Hangi dosyalar farkli?
2. Merge icin hangi dosyalar guvenli, hangileri dikkat gerektirir?
3. Cakisan modullerde hangi versiyon korunmali?
4. Eksik test coverage nerede?
5. Kritik TODO'lar hangileri?
6. Onerilen merge stratejisi nedir?

---

## RAPOR FORMATI

Raporu `docs/ARKADAS_ICIN_CLAUDE_CIKTI.md` dosyasina yaz. Format:

```markdown
# Kisi 1 Proje Tarama Raporu — [TARIH]

## 1. Git Durumu
...

## 2. Kisi 1 Alanlari (Detayli)
### strategies/
### data/
### tools/
### tests/

## 3. Kisi 2 Alanlari (Ozet)
### execution/
### bot/
...

## 4. Paylasilan Dosyalar

## 5. Execution Karsilastirma

## 6. TODO / FIXME / HACK

## 7. Import Cakismalari

## 8. Strategies Detay

## 9. Cikarim ve Oneri
```

## RAPOR YAZILDIKTAN SONRA

Raporu yazdiktan sonra:
1. `git add docs/ARKADAS_ICIN_CLAUDE_CIKTI.md`
2. `git commit -m "docs: Kisi 1 proje tarama raporu"`
3. `git push`

Boylelikle Kisi 2'nin Claude'u bu raporu alip analiz edebilir.

---

## ONEMLI NOTLAR

1. Hicbir dosyayi DEGISTIRME (rapor dosyasi haric)
2. Turkce yaz ama dosya adlari ve kod referanslari Ingilizce kalsin
3. Satir sayilarini `wc -l` ile al
4. Detayli ve dogru ol — bu rapor Kisi 2'nin Claude'u tarafindan analiz edilecek
5. `docs/CLAUDE.md` dosyasini mutlaka oku — projenin operasyonel doktrini orada
6. Eger branch'in main degilse, `git diff --stat origin/main` ile farklari goster
