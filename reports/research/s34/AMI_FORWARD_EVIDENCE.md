# AMI Forward Evidence Report

> 2026-07-03 12:21 UTC — otomatik pipeline (`ami/research/forward_pipeline.py`)

| Experiment | Knowledge | Signal | Binding | Accepted | Rejected | Dup | Forward stats |
|---|---|---|---|--:|--:|--:|---|
| E-HOUR17-FWD-001 | K-S34-HOUR17-001 | LONG_HOUR17_HOLD6H | VALID | 0 | 0 | 0 | {"n": 0} |
| E-CONVCOMP-FWD-001 | K-S34-MECH-COMPOSITE-001 | LONG_HOUR17_COMPOSITE | VALID | 0 | 0 | 0 | {"n": 0} |

Governor kararı yok (n < min_sample — birikim sürüyor).

## Kurallar (aktif)
- R1 freeze-öncesi trade = PRE_FREEZE reddi (lookahead sınırı)
- R2 spec/candidate/dataset/execution değişimi = BINDING_INVALID
- R3 trade başına tek evidence (kalıcı PK)
- R4 provenance'sız evidence reddi
- R5 pipeline izin VERMEZ — yalnız governor gate'lerine başvurur

Not: E-MECHCOMP-FWD-001 kayıtlı fakat BAĞLANMADI — shadow runner mech_score
loglamıyor (data_readiness eksik); conviction-composite için E-CONVCOMP-FWD-001 açıldı.

*Runner: `python -m ami.run_forward_pipeline` (cron/oturum başına idempotent)*