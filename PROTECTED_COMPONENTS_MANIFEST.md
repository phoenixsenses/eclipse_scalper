# PROTECTED_COMPONENTS_MANIFEST

**Tarih:** 2026-07-03 · Kaynaklar: CLAUDE.md guardrails + SYSTEM_STATE §7 + master protokol §12

## Dosya/dizin — DOKUNULMAZ (operatör sign-off olmadan)

| Bileşen | Koruma |
|---|---|
| `tools/s34_state_machine_live_executor.py` | Order logic, sizing, leverage — HİÇBİR değişiklik |
| `.env` | Okuma serbest, yazma YASAK; API key export yasak |
| `execution/`, `risk/`, `brain/` | Dizin bazında dokunma |
| Leverage (15x) / ORDER_NOTIONAL_USD / position sizing | Değişmez |
| `tools/s34_realtime_shadow_runner.py` | Shadow POLICY davranışı değişmez (observation-only loglama eklemeleri dahi operatör onayı ister — geçmişte onaylı örnek: mech_score) |
| `data/microstructure.db` | SALT-OKUNUR (`file:...?mode=ro`) |
| `start_eclipse.ps1` / `stop_eclipse.ps1` / `status_eclipse.ps1` | Proses yönetimi; restart = operatör onayı |
| Frozen prereg spec'leri (E-*-001 hash'li) | Immutable; değişiklik = yeni version + forward N sıfırlama |
| `data/ami/knowledge.sqlite` failure_archive | Rejected/falsified kayıt SİLİNMEZ |

## Çalışan prosesler — durdurma/restart operatör onayı ister

12 proses (liste: LAST_VERIFIED_CHECKPOINT.md). İnceleme read-only yapılır.

## Davranışsal sınırlar

- Observer'dan order üretilemez; dashboard canonical source olamaz.
- Min forward N'ye ulaşmak otomatik promotion değildir.
- Historical/replay forward evidence olarak kaydedilemez.
- Real liquidation ile proxy cascade aynı population'da birleştirilemez.
- Missing data ≠ 0; partial candle ≠ closed candle; swing pivot known_at_ts öncesi kullanılamaz.
- Paralel Python/PowerShell prosesi başlatılamaz (RAM); pytest ≤2 dosya + --basetemp scratchpad + -p no:cacheprovider.
