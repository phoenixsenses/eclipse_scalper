# Echo be_ratio — Information-Arrival Curve ([ts, ts+10m] dissection)

_2026-07-20T10:17:55.031963+00:00 · READ-ONLY · causal N=118 · noisy=80 · tail=14_

> Descriptive dissection of the [ts, ts+10m] contaminant window. Curve = how early the flush-continuation reveals itself. NOT a reactive threshold/gate; any overlay is FORWARD prereg. noisy defined over (ts+60s,ts+30m) so small-k co-measures it.

## When does the signal arrive? (cumulative be_ratio over [ts-10m, ts+k])

| k(min) | AUC be→noisy | AUC be→tail | AUC BTCret→tail | AUC newETHsell→tail | med BTCret bps | med newETHsell |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.550 | 0.413 | — | — | 0.0 | 0 |
| 1 | 0.634 | 0.502 | 0.214 | 0.600 | 0.2 | 1.0 |
| 2 | 0.660 | 0.525 | 0.205 | 0.637 | 1.5 | 1.0 |
| 3 | 0.683 | 0.586 | 0.210 | 0.692 | 1.4 | 1.0 |
| 4 | 0.701 | 0.630 | 0.236 | 0.678 | 0.3 | 2.0 |
| 5 | 0.703 | 0.653 | 0.205 | 0.680 | 2.0 | 2.0 |
| 6 | 0.723 | 0.714 | 0.188 | 0.708 | 3.6 | 2.0 |
| 7 | 0.728 | 0.742 | 0.158 | 0.761 | 4.4 | 2.0 |
| 8 | 0.731 | 0.730 | 0.153 | 0.755 | 5.8 | 2.0 |
| 9 | 0.734 | 0.736 | 0.152 | 0.747 | 5.3 | 2.0 |
| 10 | 0.728 | 0.729 | 0.126 | 0.753 | 5.6 | 2.0 |

## Disagreement set — late-developing flush (causal-low @k0, resolved-high @k10)

| group | n | noisy rate | tail rate | mean net | med BTCret@10m |
|---|---:|---:|---:|---:|---:|
| LATE-FLUSH (disagreement) | 16 | 0.938 | 0.312 | -40.2 | -19.2 |
| stays-low (agreement) | 43 | 0.535 | 0.047 | 63.9 | 5.8 |

## Read
- AUC be→tail rising from ~0.5 (k=0) toward ~0.73 (k=10): the k where it crosses ~0.62 is the
  minimum reactive delay a be_ratio overlay would need. Early cross => fast overlay feasible;
  only-late cross => too late to act, overlay dead.
- LATE-FLUSH vs stays-low tail/net gap = the loss a reactive cut could (forward) address —
  but at the cost of 10m delay + whipsaw on the winners in that same set. Forward-only.
