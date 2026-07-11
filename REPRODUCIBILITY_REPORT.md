# REPRODUCIBILITY_REPORT

Yalnız bir experiment/artifact yeniden üretildiğinde güncellenir.

**2026-07-03 tabanı:** Frozen prereg zinciri hash'li ve reproducible beyanlı (E-MFE50 4978edd7, E-LATENT6A a059e89d, E-LATENT6AR 1b6d0b2b, E-RISKAPP db07a737, E-BUYFADE-STRUCT 70cf5acb, E-BUYFADE-REENTRY 82a4e56b, E-BUYFADE-SILEXIT bd7d1f63). Phase 0'da yeniden-üretim koşulmadı (read-only faz).

**Phase 0 artifact reproducibility:** `QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv` deterministik üreteçten: `python tools/ami_generate_question_matrix.py` (kaynak: whitepaper App O + chart-native §23).

Şüpheli reproduksiyon kuralı: sonuç yeniden üretilemezse SİLİNMEZ; düşük reproducibility statüsüyle korunur + sebep kaydedilir (Protocol §25).
