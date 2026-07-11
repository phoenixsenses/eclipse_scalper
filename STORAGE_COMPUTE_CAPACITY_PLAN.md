# STORAGE_COMPUTE_CAPACITY_PLAN

**Tarih:** 2026-07-03. Mevcut: D: 881 GB dolu / 982 GB boş; microstructure.db 684.7 GB (büyüyor); RAM kısıtlı (paralel proses yasağının nedeni).

## Bütçeler
| Bileşen | Tavan | Aşım aksiyonu |
|---|---|---|
| canonical.sqlite (P1-3) | 5 GB | Path örnekleme çözünürlüğü düşür; operatöre bildir |
| chart objects (P4-5) | 20 GB | Sembol/TF kapsamını daralt |
| forward snapshots/paths (P8) | 50 GB/yıl | Retention sınıfları (HOT/WARM/COLD) + downsample |
| Toplam yeni AMI katmanı | 100 GB | OD ile revizyon |
| Disk boş alan alarmı | <300 GB | Yeni collector aktivasyonu dondurulur (OD-006 bağlantılı) |

## Compute/RAM
- Tek Python research prosesi aynı anda; ingest batch'leri chunk'lı; pytest ≤2 dosya.
- microstructure.db sorguları RO + indeksli + zaman-pencereli; full-scan yasak (650GB+).
- Dashboard sorguları collector'ı bloklayamaz (ayrı bağlantı, timeout'lu).

## İzleme
Her faz kapısında: DB boyutları + büyüme hızı + disk boş alan SYSTEM_STATE'e işlenir (Phase 0 tabanı bu dosyada).
