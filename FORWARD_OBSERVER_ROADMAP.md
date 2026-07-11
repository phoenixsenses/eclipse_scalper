# FORWARD_OBSERVER_ROADMAP

**Tarih:** 2026-07-03. Kural seti: frozen definition → immutable version → activation_ts → N=0 → orderless → min independent-cycle N → coverage → operatör review. Historical/replay ASLA forward sayılmaz. Her AKTİVASYON operatör onayı (master protokol §13).

## Mevcut (korunur)
| Binding | KO | Durum |
|---|---|---|
| E-HOUR17-FWD-001 | K-S34-HOUR17-001 | VALID, n=0/20 |
| E-CONVCOMP-FWD-001 | K-S34-MECH-COMPOSITE-001 | VALID, n=0/20; mech_score loglaması shadow'da canlı |

## Aday kuyruk (aktivasyon sırası önerisi; hepsi OD'li)
1. `bd_first_buy50` exit observer (OD-004) — spec hazır (Obs §17), delta-loglama, sipariş yok.
2. Silence yönetim-bilgisi forward doğrulaması (K-BUYFADE-SILENCE-INFO-001 forward kolu) — T+30 known-at etiketiyle.
3. Echo_30_90+regime paper-candidate izleme (mevcut tek paper-candidate) — mevcut shadow kayıtları üstünden observer-formalizasyonu.
4. BAD_TIMING re-entry + 4h-DOWN+silence (OD-008) — ≥6 ay veri şartı sağlanmadan AÇILMAZ.
5. Chart-native aday aileleri — yalnız Phase 6 descriptive+control geçenler, Opus D sonrası (CN §29 Phase 7).

Her aday paketi içeriği: frozen tanım + dataset hash + activation_ts + min_N (event/cycle/day) + coverage şartları + permission ceiling (max SHADOW) + zero-N honesty gösterimi.
