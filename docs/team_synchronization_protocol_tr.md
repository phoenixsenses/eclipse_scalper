# Eclipse Scalper — Takım Senkronizasyon Protokolü (Türkçe)

Bu doküman; araştırma ve execution katmanlarının sistem bütünlüğünü bozmadan nasıl çalışacağını, senkronize olacağını ve güvenli şekilde evrim geçireceğini tanımlar.

---

# 1. Temel İlke: Katman İzolasyonu

Sistem iki bağımsız katmana ayrılmıştır:

## Araştırma Katmanı (Strateji / Alfa Üretimi)

Sahibi: Araştırma mühendisi

Dizinler:

```
research/
tools/
strategies/
reports/
```

Sorumluluklar:

- Sinyal oluşturma
- Tarama (sweep) çalıştırma
- Pasif pocket'ları doğrulama
- Beklenen edge'i ölçme
- Sinyal kontratını yayımlama

Araştırma katmanı emir GÖNDERMEZ.

---

## Execution Katmanı (Emir Gönderme / Dolum Optimizasyonu)

Sahibi: Execution mühendisi

Dizinler:

```
execution/
execution/order_router.py
execution/passive_router.py
execution/fill_logic.py
execution/cancel_logic.py
execution/latency/
```

Sorumluluklar:

- Dolum oranını maksimize etme
- Olumsuz seçimi (adverse selection) minimize etme
- Emir yerleştirme zamanlamasını optimize etme
- İptal/yeniden gönderme (cancel/replace) mantığını optimize etme
- Gecikme (latency) etkisini azaltma

Execution katmanı sinyal ÜRETMEZ.

---

# 2. Sinyal Kontratı (Katmanlar Arası Arayüz)

Bu, araştırma ve execution arasındaki TEK iletişim noktasıdır.

Örnek:

```python
{
    "symbol": "ETHUSDT",
    "side": "BUY",
    "confidence": 0.82,
    "expected_edge": 0.00018,
    "max_entry_price": 2843.50,
    "timestamp": 1700000000
}
```

Araştırma bunu üretir.

Execution bunu tüketir.

Araştırma EXECUTE ETMEZ.

Execution sinyal mantığını DEGISTIRMEZ.

---

# 3. Git Branch Yapısı

Branch'ler:

```
main
research
execution
```

Araştırma mühendisi şurada çalışır:

```
research branch
```

Execution mühendisi şurada çalışır:

```
execution branch
```

Merge akışı:

```
research → main
execution → main
```

Asla doğrudan main'e commit yapılmaz.

Her zaman pull request kullanılır.

---

# 4. Günlük Senkronizasyon Protokolü

Araştırma paylaşır:

- En iyi pocket
- Beklenen edge
- Confidence dağılımı

Execution paylaşır:

- Dolum oranı (fill rate)
- Gecikme (latency)
- İptal verimliliği (cancel efficiency)

Bu metrikler sistemin ilerleme durumunu tanımlar.

---

# 5. Ortak Gerçek Dosyası (Shared Truth File)

Şu dosyayı oluştur:

```
docs/system_state.md
```

Örnek:

```
ACTIVE_STRATEGY: micro_edge_v3

BEST_POCKET:
imbalance >= 0.50
intensity >= 2500
spread <= 0.0005

EXPECTED_EDGE: 0.000015
FILL_RATE: 57%
TARGET_FILL_RATE: 70%
```

Bu dosya mevcut üretim gerçeğini temsil eder.

---

# 6. Sorumluluk Matrisi

Araştırma Mühendisi:

- Özellik mühendisliği (feature engineering)
- Sinyal doğrulama
- Geriye dönük test (backtesting)
- Tarama optimizasyonu
- Pocket keşfi

Execution Mühendisi:

- Emir yerleştirme optimizasyonu
- Dolum oranı optimizasyonu
- İptal mantığı optimizasyonu
- Gecikme azaltma

---

# 7. Merge Öncesi Test Protokolü

Zorunlu komutlar:

```bash
pytest

python -m tools.validate_passive_pocket_forward

python -m tools.rank_passive_pockets_forward
```

Execution değişikliklerinde ayrıca şu da çalıştırılmalıdır:

```bash
python -m execution.bootstrap
```

---

# 8. Katı Ayrım Kuralı

Yanlış (yasak):

```python
place_order()
```

Doğru:

Araştırma:

```python
emit_signal()
```

Execution:

```python
consume_signal()
```

---

# 9. Günlük Çalışma Akışı

Araştırma mühendisi:

```
sweep çalıştır
forward validation çalıştır
en iyi pocket'ı güncelle
research branch'e commit yap
```

Execution mühendisi:

```
order routing'i optimize et
fill rate'i ölç
execution branch'e commit yap
```

Doğrulama sonrası merge yapılır.

---

# 10. Sistem Modeli: Üretici / Tüketici

Araştırma katmanı:

Üretici (Producer)

Execution katmanı:

Tüketici (Consumer)

Araştırma alfa üretir.

Execution alfayı gerçekleşmiş kâra dönüştürür.

---

# 11. Hedef

Belirleyici (deterministic), ölçeklenebilir ve üretim kalitesinde pasif bir trading sistemi inşa etmek; burada:

- Alfa üretimi bağımsız çalışır
- Execution bağımsız olarak optimize edilir
- Sistem regresyon olmadan güvenle evrim geçirir

Bu mimari, kurumsal düzeyde güvenilirlik ve ölçeklenebilirlik sağlar.

---

DOKÜMAN SONU
