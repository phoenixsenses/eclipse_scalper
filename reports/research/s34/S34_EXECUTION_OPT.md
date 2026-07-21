# S34 Execution Optimizer (Faz 4)

> gated event=97, TRAIN/TEST 70/30. 2026-07-02

- **E1_mark**: N=96 WR=60.4% avg=+30.3 worst=-349.5 mc=0.01
- **E1_ask_bid**: N=96 WR=60.4% avg=+30.9 worst=-349.8 mc=0.01 EV/sig=30.9 fill=None%
- **E2_limit10**: N=64 WR=67.2% avg=+36.3 worst=-190.9 mc=0.008 EV/sig=24.2 fill=66.7%
- **E2_limit20**: N=45 WR=68.9% avg=+40.1 worst=-178.4 mc=0.012 EV/sig=18.8 fill=46.9%
- **E2_limit30**: N=32 WR=62.5% avg=+44.9 worst=-161.3 mc=0.014 EV/sig=15.0 fill=33.3%
- **E3_vwap5**: N=97 WR=58.8% avg=+30.8 worst=-347.3 mc=0.006

## TP x SL (TRAIN-sirali top5, TEST raporu)

- tp=fix300 sl=fix300: TEST N=30 WR=70.0% avg=+33.1 worst=-305.0 mc=0.114
- tp=fix300 sl=none: TEST N=30 WR=70.0% avg=+47.5 worst=-150.8 mc=0.018
- tp=fix200 sl=fix300: TEST N=30 WR=70.0% avg=+27.3 worst=-305.0 mc=0.144
- tp=fix200 sl=none: TEST N=30 WR=70.0% avg=+41.8 worst=-150.8 mc=0.012
- tp=none sl=fix300: TEST N=30 WR=70.0% avg=+36.2 worst=-305.0 mc=0.11

- baseline(none,none): TEST N=30 avg=50.7 worst=-150.8

---
*Script: tools/s34_execution_optimizer.py*