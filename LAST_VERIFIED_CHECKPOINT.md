# LAST_VERIFIED_CHECKPOINT

**Belirlendi:** 2026-07-03, Fable 5 Phase 0 audit
**Protokol:** `docs/protocols/AMI_S34_MASTER_EXECUTION_PROTOCOL_v1.1.md`

## Checkpoint kimliği

| Alan | Değer |
|---|---|
| Git HEAD | `5cda3122` — "Fix collector WS fallback: invalid Binance stream URL + merge overall health" |
| Branch | `codex/data-layer-fallback-cleanup` |
| SYSTEM_STATE son bölüm | §39 — BUY-FADE Silence-Conditional Exit Timing (2026-07-03) |
| AMI test suite | **119/119** (17 core + 20 mutation + 15 latent + 14 regime + 13 risk + 24 buyfade-struct + 16 silexit) |
| Whitepaper patch seviyesi | PATCH-0007 (v0.2 canonical zinciri 0.2.7) |
| Decision Records | DR-0001..DR-0007 |
| AMI changelog | AMI-CHG-0007 |

## Doğrulanmış runtime durumu (2026-07-03 audit anı)

- **12 Python prosesi canlı** (PID'ler audit anı): orderflow_chart 19288, s34_replay 19744, collector_supervisor 24472, heartbeat_watchdog 22108, microstructure_collector 2296, bookticker_collector 24692, oi_spot_poller 23772, s34_shadow_paper_runner 19292, s34_live_chart 9268, s34_v_engine_v02_shadow_mirror 22428, s34_realtime_shadow_runner 25944, event_diary 19824.
- **Live executor ÇALIŞMIYOR** (`-EnableLive` default KAPALI — beklenen durum).
- `data/microstructure.db` son yazma 2026-07-03 20:23 (collector canlı); 684.7 GB.
- `data/s34_intelligence.db` son yazma 2026-07-03 18:57.
- AMI store'lar: `knowledge.sqlite` (11 KO, 21 failure-archive, 40 audit), `research.sqlite` (14 soru [slug-ID], 10 deney, 8 evidence, **2 forward binding VALID**: E-HOUR17-FWD-001 + E-CONVCOMP-FWD-001, n=0/20), `decisions.jsonl`.
- Shadow ledger: `reports/shadow/s34_state_machine_shadow.jsonl` mevcut.
- Disk: D: 881 GB dolu / 982 GB boş.

## Uncommitted çalışma (git status)

- Modified: `tools/s34_cascade_navigation_dashboard.py`, `tools/s34_realtime_shadow_runner.py` (önceki oturumlardan, observation-only).
- Untracked: 5 canonical belge + CLAUDE.md + SYSTEM_STATE.md + ami/ + data/ami/ + docs/ami/ + docs/protocols/ + tüm S34 rapor katmanı. **Hiçbiri commit edilmemiş** — versiyon kontrol riski (bkz. RISK_REGISTER R-16).

## Son doğrulanmış bilimsel durum (özet)

SYSTEM_STATE §33–§39: AMI Faz 0-6A tamam; 6A latent REJECTED (dürüst null), 6A-R rejim PASS (trend=UP, max SHADOW), 6A-R2 risk FALSIFIES/INSUFFICIENT, BUYFADE struct/reentry/silexit FALSIFIES×3 + K-BUYFADE-SILENCE-INFO-001 (HOLDOUT_VALIDATED, max SHADOW). Hiçbir bileşen LIVE izne sahip değil.

## En erken eksik dependency

Master protokol Phase 1 (Canonical reconciliation): **canonical warehouse + numeric question registry + contamination/exposure ledger yok**. İlk güvenli batch: `FIRST_SAFE_IMPLEMENTATION_BATCH.md`.
