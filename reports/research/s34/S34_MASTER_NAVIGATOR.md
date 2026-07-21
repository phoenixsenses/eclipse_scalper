# S34 Master Navigator — conviction-priority scheduler

> Route havuzu 193 event {'LONG_comp': 85, 'LONG_100k': 97, 'SHORT_1317': 11}, 4.5 ay. LOOKAHEAD YOK. Tarih 2026-07-01

| Politika | N | /ay | WR | avg | TOTAL | mdd | mc_p | mix |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| P0_long_comp_only | 42 | 9.3 | 73.8% | +59 | +2462 | -176 | 0.0 | {'LONG_comp': 42} |
| P1_fifo_union | 60 | 13.2 | 70.0% | +61 | +3654 | -264 | 0.0 | {'LONG_comp': 17, 'LONG_100k': 39, 'SHORT_1317': 4} |
| P2_admit_ge4 | 47 | 10.4 | 66.0% | +64 | +2985 | -266 | 0.0 | {'LONG_comp': 13, 'LONG_100k': 30, 'SHORT_1317': 4} |
| P2_admit_ge5 | 32 | 7.1 | 71.9% | +59 | +1881 | -227 | 0.014 | {'LONG_comp': 8, 'LONG_100k': 20, 'SHORT_1317': 4} |
| P3_route_priority | 49 | 10.8 | 73.5% | +58 | +2857 | -222 | 0.0 | {'LONG_comp': 31, 'LONG_100k': 14, 'SHORT_1317': 4} |
| P4_route_priority_strict | 33 | 7.3 | 66.7% | +56 | +1851 | -353 | 0.01 | {'LONG_comp': 22, 'LONG_100k': 7, 'SHORT_1317': 4} |
| P3_route_priority_weighted | 49 | 10.8 | 73.5% | +58 | +2857 | -222 | 0.0 | {'LONG_comp': 31, 'LONG_100k': 14, 'SHORT_1317': 4} |
| P2_admit_ge4_weighted | 47 | 10.4 | 66.0% | +64 | +2985 | -266 | 0.0 | {'LONG_comp': 13, 'LONG_100k': 30, 'SHORT_1317': 4} |

**Conviction-weighted sizing:**
- P3_route_priority_weighted: flat=2857.0 weighted=7155.0 per_unit=55.5
- P2_admit_ge4_weighted: flat=2985.0 weighted=8730.0 per_unit=62.8

---
*Script: tools/research_s34_master_navigator.py*