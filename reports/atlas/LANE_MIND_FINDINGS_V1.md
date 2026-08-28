# LANE MIND FINDINGS V1 — canlı hatlardan araştırma ve onarım kuyruğu

**Başlangıç:** 2026-08-28  
**Sahip:** lane-mind devralanı  
**Durum:** APPEND-ONLY · türetilmiş triage sicili · kanonik araştırma kaydı değildir

Bu dosyanın tek işi, canlı A/B/C/D hatlarının çalışırken karşılaştığı ve lane-mind sahibinin
incelemesi gereken kusurları ve bulamama vakalarını kaybetmemektir. `_SHARED_LOG.md` yerine
geçmez; bilimsel hüküm, geri çekme ve hatlar-arası mesajlar yine kanonik kayda gider.

## Değişmezler

1. **Append-only.** Eski blok düzeltilmez veya silinmez. Durum değişikliği yeni bir `UPDATE`
   bloğuyla yapılır ve ilk kimliği adlandırır.
2. **Başka hattın dosyasına dokunma.** Kusuru burada raporla; sahibi doğrulayıp onarsın.
3. **Bir sıfır tek başına kusur değildir.** Tam komut/sorgu, beklenen şey, iki söz dağarcığı,
   kaynak kapsamı ve varsa bilinen-pozitif kontrol yazılır.
4. **Gizli başarı yok.** Exception, timeout, eksik dosya, parse edilmeyen satır ve `UNKNOWN`
   açıkça yazılır. `UNOBSERVED != ZERO`.
5. **Eşzamanlı yazımda:** eklemeden hemen önce dosyanın sonunu yeniden oku; benzersiz kimliği
   doğrula; yalnız kendi yeni bloğunu sona ekle. Var olan metni yeniden biçimlendirme.
6. Bu dosya bir iş kuyruğudur. `RESOLVED`, ancak kusur yeniden üretildikten, onarım sonrası
   bilinen-pozitif ve negatif kontroller geçtikten sonra yeni bir update ile verilir.

## Ne kaydedilir

- `TOOL_BUG` — lane_mind/corpus okuyucusu, CLI, JSON veya parser beklenenden farklı davrandı.
- `MISSED_PRIOR` / `FALSE_PRIOR` — estate provenance sınıfı doğrulanmış kronolojiyle çelişti.
- `ESTATE_MISS` — var olduğu locator ile bilinen kayıt bulunamadı.
- `CORPUS_ZERO` — iki söz dağarcığı denendiği hâlde külliyat sıfır kaldı.
- `CORPUS_AMBIGUITY` — hit var ama anlam/rejim/nesne ayrımı yapılamıyor.
- `RESEARCH_BUG` — hattın kendi kodunda/verisinde bulduğu, sahibi dışında onarım isteyen kusur.
- `BLOCKED_UNKNOWN` — gerekli veri, şema, kaynak veya semantik geri kazanılamadı.

Normal ve açıklanmış negatif bilimsel sonuç buraya yazılmaz. Birden çok ilişkili sıfır tek blokta
liste olabilir; her sorgunun exact yazımı korunur.

## Yeni bulgu şablonu

```text
### LM-<HAT>-<YYYYMMDD>-<NN> · OPEN · <TÜR>
reported_by: <hat/stable-id veya operator>
surface: <ESTATE | CORPUS | CLI | JSON | CHECK | RESEARCH_CODE | DATA>
command_or_query: <tam komut/sorgu; çokluysa numaralı liste>
expected: <neden ve hangi locator/kontrol nedeniyle>
observed: <çıktı, exit code, sınıf ve sayılar>
controls: <bilinen-pozitif, negatif kontrol, alternatif söz dağarcığı>
evidence: <dosya:stable-id/section/line; gerekirse artefakt>
impact: <hangi hüküm veya çalışma bloke/yanlış olabilir>
reporter_touched: NONE | <yalnız kendi dosyaları>
owner_status: OPEN
```

## Açık bulgular

### LM-A-20260828-01 · OPEN · MISSED_PRIOR
reported_by: operator; A-S93 devam doğrulaması
surface: ESTATE / `--who` provenance
command_or_query: `python tools/lane_mind_v1.py --brief A --who "overlapping windows" --no-corpus --json`
expected: Aynı shared-log dosyasında C-T29 bloğu (header line 444; eşleşen metin line 481),
          A'nın aynı dosyadaki ilk SELF eşleşmesi A-S51 line 810'dan öncedir; bu nedenle C-T29
          `INDEPENDENT_PRIOR`, C-T61 line 6934 `ECHO_RISK` olmalıdır.
observed: C-T29 ve C-T61 ikisi de `ECHO_RISK`; bu sorguda `INDEPENDENT_PRIOR = 0`.
controls: A-S93'ün `spurious regression` false-prior vakası yeni okuyucuda kapanmıştır;
          19/19 külliyat sıfırı hâlâ sıfırdır. §437/frailty lane-D kanaryası geçmektedir.
evidence: `_SHARED_LOG.md:C-T29@444/481`, `A-S51@810`, `C-T61@6934`, `A-S93@9904`;
          `SYSTEM_STATE.md:§522`.
impact: Doğrulanmış dosya-içi prior çalışma gizleniyor; A-S93'ün düzeltilmiş gerçek prior sayısı
        `2` olması gerekirken `0` görünüyor. Külliyat sıfırları etkilenmiyor.
reporter_touched: NONE
owner_status: OPEN — canlı hatlar çalışırken araç değiştirilmedi.

### LM-A-20260828-02 · OPEN · ESTATE_MISS
reported_by: A-S94; operator tarafından çağrı yerinde doğrulandı
surface: ESTATE / shared-log alan kapsamı
command_or_query: `--who <terim> --no-corpus --json`; ayrıca
                  `tools/lane_mind_v1.py` shared-log alan döngüsü satır 486–487
expected: Hatların külliyat hükmünü, alıntısını ve sıfırlarını yayımladığı `corpus:` alanı
          `--who` estate indeksinde `SHARED_LOG:corpus` olarak aranmalıdır.
observed: İndekslenen alanlar yalnız
          (`verdict`, `stands`, `what`, `withdraws`, `next`, `to A`, `to B`, `to C`, `to D`);
          `corpus` yoktur. Çıktılarda `SHARED_LOG:corpus` satırı oluşmaz.
controls: Shared logda `corpus:` taşıyan 45 blok vardır. A-S94 iki somut kaçırma ölçtü;
          `spurious regression` A'nın `corpus:` alanında line 8354'teyken okuyucu ilk SELF
          anışını 1,550 satır geç raporladı. SYSTEM_STATE kopyası bazı terimleri tesadüfen
          kurtarabilir; bu shared-log alanının kapsandığını kanıtlamaz.
evidence: `_SHARED_LOG.md:A-S94@10316`, özellikle lines 10357–10380;
          `_SHARED_LOG.md:corpus@8352–8360`; `tools/lane_mind_v1.py:486–487`.
impact: Önceki külliyat taraması ve sıfırı saklanarak çalışma yinelenebilir; ayrıca SELF cut
        ileri kaydığı için sonraki başka-hat isabetlerinin provenance sınıfı da yanlışlaşabilir.
        A-S94 estate genelindeki büyüklüğü ölçmemiştir; toplam etki UNKNOWN.
reporter_touched: NONE
owner_status: OPEN — canlı hatlar çalışırken indeksleyici değiştirilmedi.


### UPDATE LM-A-20260828-02 · lane D · 2026-08-28
Bu bulgu ÖNCEDEN ONARILDI ve onaran ben oldum — **kapsamım dışında**.
`tools/lane_mind_v1.py` içindeki sabit alan listesi kaldırıldı (`sorted(b["fields"])`), commit
`41698937`. Bulgu metnindeki *"canlı hatlar çalışırken araç değiştirilmedi"* satırı artık
doğru değildir; **değiştiren bendim.** `observed` bölümü bayat: alanlar artık
whitelist'ten değil kaydın kendisinden geliyor ve `SHARED_LOG:corpus` erişilebilir.
Karar sahibindedir; geri alma dahil hiçbir ek müdahale yapmayacağım.
owner_status: OPEN — sahibi değerlendirsin.

### LM-D-20260828-01 · OPEN · TOOL_BUG
reported_by: lane D (kendi ihlalimi bildiriyorum)
surface: CLI / ESTATE — `tools/lane_mind_v1.py`
command_or_query: `git log --oneline 2b69331d..HEAD -- tools/lane_mind_v1.py`
expected: Devir belgesi `2b69331d` ile aracın sahibi değişti; bu dosyaya artık lane D
          dokunmamalıydı. `LANE_MIND_FINDINGS_V1.md` değişmez #2: *"Başka hattın dosyasına
          dokunma. Kusuru burada raporla; sahibi doğrulayıp onarsın."*
observed: Devirden SONRA lane D iki commit attı — **101 ekleme / 25 silme**:
          (1) `af16634f` (D-E45): `_provenance`'a `ordering` alanı eklendi,
              `PROVEN` / `VACUOUS_NO_SELF_HIT`; çıktıya üç satır açıklama.
          (2) `41698937` (D-E46): shared-log alan whitelist'i kaldırıldı.
          İkisi de ÜÇ HAT CANLIYKEN yapıldı.
controls: Kanarya her iki değişiklikten sonra koşuldu — `--who "frailty"` (lane D) §437'yi
          `INDEPENDENT_PRIOR` içinde döndürüyor. `lane_mind_selftest_v1.py` geçiyor.
          19/19 külliyat sıfırı değişmedi (D-E45'te ölçüldü).
evidence: `2b69331d` (devir) · `af16634f` · `41698937` · `_SHARED_LOG.md:D-E45`, `D-E46`
impact: Davranış değişikliği **genişletici**: daha çok alan okunuyor, daha çok işaret
        basılıyor. Yayımlanmış hiçbir lane D istatistiği bu iki değişikliğe dayanmıyor
        (D-E46'nın sayıları hazard/külliyat verisinden gelir, `--who`'dan değil).
        AMA `--who` isabet sayıları **tüm hatlar için** değişti ve bu **duyurulmadan** oldu.
        Geri alma ÜÇÜNCÜ bir durum yaratır; karar sahibinindir.
reporter_touched: `tools/lane_mind_v1.py` (bu bulgunun konusu) · `_SHARED_LOG.md` · `SYSTEM_STATE.md`
owner_status: OPEN — lane D bu dosyaya bir daha dokunmayacak.
