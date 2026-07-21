# AMI MFE+50 Experiment — Baseline vs Candidate

> 2026-07-02 — prereg FROZEN hash `4978edd7238e6acb` (hesaplamadan once). Milestone N=42, etiketler: {'continuation': 29, 'negative': 5, 'stall': 8}

**TRAIN kural secimi:** frozen protokol (tek-feature medyan split, kisit: TRAIN continuation_capture>=0.85) — **HICBIR KURAL KISITI SAGLAYAMADI (best=None)** -> politika baselinea cokuyor, ayirma hipotezi TRAIN asamasinda dustu.

## TEST (untouched %30) — kontrol politikalariyla

| Politika | N | WR | median | mean | cum | top3-rm | PF | maxDD | MFE | MAE | give% | give-kurtarma% | cont-capture |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| HOLD_baseline | 13 | 84.6% | 93.1 | 101.0 | 1313.0 | 546.0 | 15.3 | -46.9 | 172.2 | -69.8 | 15.4% | 0.0 | 1.0 |
| POLICY_flagged | 13 | 84.6% | 93.1 | 101.0 | 1313.0 | 546.0 | 15.3 | -46.9 | 172.2 | -69.8 | 15.4% | 0.0 | 1.0 |
| LOCK_ALL_control | 13 | 100.0% | 20.0 | 92.7 | 1205.0 | 463.0 | None | 0.0 | 172.2 | -69.8 | 15.4% | 100.0 | 0.75 |
| EXIT_ALL_control | 13 | 100.0% | 45.0 | 45.0 | 585.0 | 450.0 | None | 0.0 | 172.2 | -69.8 | 15.4% | 100.0 | 0.0 |

- Flag orani (TEST): 0.0%
- Session breakdown (policy): {"US": {"n": 12, "mean": 95.1}, "OFF": {"n": 1, "mean": 171.3}}
- Execution feasibility: cikislar mark-fill; E1 bulgusu geregi mark~ask/bid (~0.6bps).

## SONUC: **FALSIFIES**

Yorum: +50 aninda mevcut 10 tek-featurein hicbiri, continuationlari korurken givebackleri
ayiramadi. Kontroller de teyit ediyor: LOCK_ALL cum 1205 < HOLD 1313 (givebackleri kurtariyor
ama continuationdan calarak); EXIT_ALL cum 585 (edgei kesiyor). **HOLD baseline ayakta.**
Failure archive kaydi: retry = post-entry state-TRANSITION dizileri + cift-feature (YENI prereg ile).

Durust statu: software-correct OK · replay-validated OK · holdout-validated N/A (aday yok) ·
forward-validating N/A · operationally-permitted YOK.

*Script: tools/research_ami_mfe50_experiment.py — prereg: E-MFE50-001 — evidence: EV-MFE50-001 (FALSIFIES)*