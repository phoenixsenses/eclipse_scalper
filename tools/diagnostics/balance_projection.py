"""
2-month balance projection based on live paper performance.
Monte Carlo simulation: 10,000 paths.
"""
import random
import statistics
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Live paper stats (ETH_500K + SOL_200K_CLEAN only) ──────────────────────
# From 10.3-day live paper observation
# Observed frequency
ETH_FREQ_PER_DAY  = 1.843   # trades/day
SOL_FREQ_PER_DAY  = 0.970   # trades/day (clean geo only)
DAYS              = 60       # 2 months

# Per-trade outcome distributions (from live paper, simplified to TP/SL/BE)
# ETH: WR=79%, TP=60bps, SL=40bps, BE~0  — net cost ~3bps fees per leg
ETH_OUTCOMES = [
    (0.79, +57),   # TP hit net (~60 - fees)
    (0.06, -3),    # BE hit net (~0 - fees)
    (0.15, -43),   # SL hit net (-40 - fees)
]
# SOL clean: WR=80%, TP=60bps, SL=30bps
SOL_OUTCOMES = [
    (0.80, +57),
    (0.05, -3),
    (0.15, -33),
]

LEVERAGE      = 40
MARGIN_PCT    = 0.85   # of balance
SIM_PATHS     = 10_000
START_BALANCE = 35.0
MIN_BALANCE   = 5.0    # below this can't place orders

def draw_outcome(outcomes):
    r = random.random()
    cum = 0.0
    for prob, bps in outcomes:
        cum += prob
        if r <= cum:
            return bps
    return outcomes[-1][1]

def run_path(eth_trades_day, sol_trades_day, days, start_bal):
    balance = start_bal
    for _ in range(days):
        # ETH trades this day (Poisson)
        n_eth = 0
        for _ in range(int(eth_trades_day * 3 + 1)):
            if random.random() < eth_trades_day / (eth_trades_day * 3 + 1):
                n_eth += 1
        n_eth = max(0, round(random.gauss(eth_trades_day, eth_trades_day**0.5)))

        n_sol = max(0, round(random.gauss(sol_trades_day, sol_trades_day**0.5)))

        for _ in range(n_eth):
            if balance < MIN_BALANCE:
                break
            margin = balance * MARGIN_PCT
            notional = margin * LEVERAGE
            bps = draw_outcome(ETH_OUTCOMES)
            pnl = notional * bps / 10000
            balance += pnl

        for _ in range(n_sol):
            if balance < MIN_BALANCE:
                break
            margin = balance * MARGIN_PCT
            notional = margin * LEVERAGE
            bps = draw_outcome(SOL_OUTCOMES)
            pnl = notional * bps / 10000
            balance += pnl

    return max(balance, 0.0)


def scenario(label, eth_freq, sol_freq):
    random.seed(42)
    results = [run_path(eth_freq, sol_freq, DAYS, START_BALANCE) for _ in range(SIM_PATHS)]
    results.sort()
    p10  = results[int(0.10 * SIM_PATHS)]
    p25  = results[int(0.25 * SIM_PATHS)]
    p50  = results[int(0.50 * SIM_PATHS)]
    p75  = results[int(0.75 * SIM_PATHS)]
    p90  = results[int(0.90 * SIM_PATHS)]
    mean = statistics.mean(results)
    bust = sum(1 for r in results if r < MIN_BALANCE) / SIM_PATHS * 100
    print(f"\n{label}")
    print(f"  p10={p10:>8.0f}  p25={p25:>8.0f}  median={p50:>8.0f}  p75={p75:>8.0f}  p90={p90:>8.0f}  mean={mean:>10.0f}  bust={bust:.1f}%")


print(f"Monte Carlo — {SIM_PATHS:,} paths, start={START_BALANCE} USDT, {DAYS} days")
print(f"Margin: {MARGIN_PCT*100:.0f}% x {LEVERAGE}x leverage = {MARGIN_PCT*LEVERAGE:.0f}x notional")
print(f"Columns: p10 / p25 / median / p75 / p90 / mean / bust%")

scenario("OPTIMISTIC  (observed rate: ETH=1.84/d, SOL=0.97/d)",
         ETH_FREQ_PER_DAY, SOL_FREQ_PER_DAY)

scenario("BASE CASE   (half rate:     ETH=0.92/d, SOL=0.49/d)",
         ETH_FREQ_PER_DAY * 0.5, SOL_FREQ_PER_DAY * 0.5)

scenario("CONSERVATIVE(quarter rate:  ETH=0.46/d, SOL=0.24/d)",
         ETH_FREQ_PER_DAY * 0.25, SOL_FREQ_PER_DAY * 0.25)

scenario("WORST CASE  (rare signals:  ETH=0.20/d, SOL=0.10/d)",
         0.20, 0.10)

print()
print("SL STREAK STRESS TEST (starting $35, ETH SL=40bps, 85%x40x)")
bal = START_BALANCE
loss_factor = 1 - (MARGIN_PCT * LEVERAGE * 40 / 10000)
for n in range(1, 8):
    bal *= loss_factor
    print(f"  {n} consecutive SL: ${bal:.2f}")
