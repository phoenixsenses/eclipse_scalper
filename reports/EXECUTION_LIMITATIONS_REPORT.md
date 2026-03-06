# EXECUTION_LIMITATIONS_REPORT

## Current model limitations found in code

1. `tools/micro_edge_backtest.py` fixed-cost models (`taker`, `maker`, `mid`, `halfspread`) apply deterministic per-trade cost and assume every qualified signal fills. This overstates attainable participation and understates miss/cancel risk.
2. Previous `maker` path used static `maker_penalty_bps`, without conditioning on spread/intensity/volatility/imbalance or order queue state.
3. No queue-priority model existed (no place-in-queue, no expected queue depletion, no cancellation priority effects).
4. No explicit no-fill state existed in fixed models; all events became trades, biasing realized capacity and turnover.
5. No partial-fill mechanics existed in fixed models; they implicitly assumed full size execution at one synthetic price.
6. No event-time fill window / timeout logic existed in fixed models, so stale opportunities were incorrectly assumed executable.
7. Adverse selection was previously represented by one static penalty, not data-conditioned post-touch behavior.
8. Sweep tooling ranked by net metrics under these simplified assumptions, so model risk dominated any signal-edge claims.

## Why this biases results

- It can overestimate practical fill rates for passive intents and under-model missed alpha due to queueing.
- It can misprice cost variance across regimes; high-vol/high-intensity periods need higher expected adverse selection.
- It prevents separating signal quality from execution feasibility, which is essential for deployment gating.
