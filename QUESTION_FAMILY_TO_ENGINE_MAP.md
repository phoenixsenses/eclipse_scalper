# QUESTION_FAMILY_TO_ENGINE_MAP

**Tarih:** 2026-07-03 · CSV: `QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv` (1058 satır; 192 verbatim, 866 MISSING_CANONICAL_TEXT)

Ortak kontrol üreteçleri (HER aile): matched non-event controls, event-never-arrived controls, random-time control, opposite-direction control, WAIT/NO_TRADE benchmark, cycle-grouped split + purge/embargo, family-level multiple-testing.

| Aile (canonical parent) | Q aralığı | Ortak engine | Ana data dependency | Phase | Not |
|---|---|---|---|---|---|
| BUYFADE_EVIDENCE_STRUCTURAL_THESIS | Q001–024 | mevcut buyfade suite (E-BUYFADE-*) | microstructure.db (RO) | 6 | Büyük kısmı 2026-07-03 verdict'leriyle CEVAPLI — warehouse seed'de map edilecek |
| PRE_EVENT_LONG_GENESIS_MATURITY | Q025–085 | LONG-genesis engine (yeni) + all-timestamp candidate universe | P3 identity + §17.8 universe | 6 | Universe yoksa BLOCKED_BY_DATA |
| SHORT_ENTRY_TIMING | Q086–147 | entry-timing engine (delay/price/flow grid — buyfade B varyantı genelleştirilir) | P3 + P5 | 6 | |
| SILENCE_ONSET_MATURITY_BREAKDOWN | Q148–175 | silence engine (mevcut silence_v1 known-at'li) | mevcut + P2 kontrat | 6 | Silence=YÖNETİM bilgisi verdict'i taban |
| SHORT_HORIZON_EXIT_MANAGEMENT | Q176–218 | management/exit engine (fixed/structural/partial grid) | P3 paths | 7 | |
| STOP_TAXONOMY_SHORT_REENTRY | Q219–243 | stop-taxonomy + re-entry engine | P3 paths + shadow ledger RO | 7 | S→S churn FALSIFIED; BAD_TIMING alt-sinyali OD-008 |
| LONG_SHORT_TRANSITIONS | Q244–299 | transition engine (reclaim→LONG vb.) | P3 cycle states | 7 | |
| MULTI_TF_REGIME_OPPOSITE_LIQ | Q300–335 | multi-TF matrix engine (ami/states üstünde) | P3 + states | 6 | 1W hücreleri INSUFFICIENT (veri<20 hafta) |
| REPLICATION_FORWARD_GOVERNANCE | Q336–395 | governance/reproducibility engine | P1 warehouse | 2 | Registry inşasıyla birlikte |
| POSITION_AWARE_CYCLE_PATH_MECHANISM | Q396–534 | position-aware action-value engine + path taxonomy | P3 (kritik) | 3→7 | |
| SIGNAL_AGING_CLOCK_ROUTE_HOLD_EXECUTION | Q535–730 | aging/market-clock/competing-risk/execution engines | P3 + execution telemetry (§17.6) | 7 | Telemetry eksikse BLOCKED_BY_DATA |
| EVIDENCE_INDEPENDENCE_CAUSAL_OOD_META | Q731–866 | epistemic engines (contamination/OOD/meta) | P2 ledger'lar | 2→10 | OOD/kalibrasyon kısmı Phase 10 |
| CHART: Candle/Close Morphology | Q867–878 | candle-morphology engine | P4 candle objects | 4→6 | |
| CHART: Push/Momentum Geometry | Q879–890 | push-geometry engine | P4 push objects | 4→6 | |
| CHART: Swing Grammar | Q891–902 | swing-grammar engine | P4 confirmed swings | 4→6 | |
| CHART: Sweep Anatomy | Q903–914 | sweep engine | P4 level+swing | 4→6 | |
| CHART: Breakout/Retest | Q915–926 | breakout engine | P4 levels | 4→6 | |
| CHART: Compression | Q927–938 | compression engine | P4 | 4→6 | |
| CHART: Trendline/Channel | Q939–950 | channel engine | P4 | 4→6 | |
| CHART: Relative Strength | Q951–962 | RS/lead-lag engine | çoklu-sembol veri | 4→6 | |
| CHART: Session/Opening | Q963–974 | session-structure engine | P4 | 4→6 | |
| CHART: Unconditional SHORT Genesis | Q975–986 | SHORT-genesis engine (LONG-genesis simetriği) | P3+P4+universe | 6 | |
| CHART: Setup Cancellation | Q987–998 | setup-lifecycle engine | P4 setup FSM | 7 | |
| CHART: Human Observation Registry | Q999–1010 | observation-bridge (timestamp+screenshot provenance) | P4 registry | 7 | İnsan girişi governance'lı (OD gerektirir) |
| CHART: Volume/Participation | Q1011–1022 | volume-coupling engine | bookticker+trades | 4→6 | |
| CHART: Auction Inefficiency | Q1023–1034 | inefficiency/repair engine | P4 | 6 | |
| CHART: Multi-TF Visual Nesting | Q1035–1046 | nesting engine (states+P4) | P4+states | 6 | |
| CHART: Validation/Evidence Independence | Q1047–1058 | epistemic engines (paylaşımlı) | P2 | 2→10 | |
