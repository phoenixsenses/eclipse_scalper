# Eclipse Scalper — Team Synchronization Protocol

This document defines how research and execution layers operate, synchronize, and evolve safely without breaking system integrity.

---

# 1. Core Principle: Layer Isolation

The system is divided into two independent layers:

## Research Layer (Strategy / Alpha Generation)

Owned by: Research engineer

Directories:

```
research/
tools/
strategies/
reports/
```

Responsibilities:

- Build signals
- Run sweeps
- Validate passive pockets
- Measure expected edge
- Publish signal contract

Research layer does NOT place orders.

---

## Execution Layer (Order Placement / Fill Optimization)

Owned by: Execution engineer

Directories:

```
execution/
execution/order_router.py
execution/passive_router.py
execution/fill_logic.py
execution/cancel_logic.py
execution/latency/
```

Responsibilities:

- Maximize fill rate
- Minimize adverse selection
- Optimize order placement timing
- Optimize cancel/replace logic
- Reduce latency impact

Execution layer does NOT generate signals.

---

# 2. Signal Contract (Interface Between Layers)

This is the ONLY communication point between research and execution.

Example:

```python
{
    "symbol": "ETHUSDT",
    "side": "BUY",
    "confidence": 0.82,
    "expected_edge": 0.00018,
    "max_entry_price": 2843.50,
    "timestamp": 1700000000
}
```

Research produces this.

Execution consumes this.

Research does NOT execute.

Execution does NOT modify signal logic.

---

# 3. Git Branch Structure

Branches:

```
main
research
execution
```

Research engineer works on:

```
research branch
```

Execution engineer works on:

```
execution branch
```

Merge flow:

```
research → main
execution → main
```

Never commit directly to main.

Always use pull requests.

---

# 4. Daily Synchronization Protocol

Research shares:

- best pocket
- expected edge
- confidence distribution

Execution shares:

- fill rate
- latency
- cancel efficiency

These metrics define system progress.

---

# 5. Shared Truth File

Create file:

```
docs/system_state.md
```

Example:

```
ACTIVE_STRATEGY: micro_edge_v3

BEST_POCKET:
imbalance >= 0.50
intensity >= 2500
spread <= 0.0005

EXPECTED_EDGE: 0.000015
FILL_RATE: 57%
TARGET_FILL_RATE: 70%
```

This file represents current production truth.

---

# 6. Responsibility Matrix

Research Engineer:

- Feature engineering
- Signal validation
- Backtesting
- Sweep optimization
- Pocket discovery

Execution Engineer:

- Order placement optimization
- Fill rate optimization
- Cancel logic optimization
- Latency reduction

---

# 7. Test Protocol Before Merge

Required commands:

```
pytest

python -m tools.validate_passive_pocket_forward

python -m tools.rank_passive_pockets_forward
```

Execution changes must also run:

```
python -m execution.bootstrap
```

---

# 8. Strict Separation Rule

Incorrect (forbidden):

```python
place_order()
```

Correct:

Research:

```python
emit_signal()
```

Execution:

```python
consume_signal()
```

---

# 9. Daily Workflow

Research engineer:

```
run sweep
run forward validation
update best pocket
commit research branch
```

Execution engineer:

```
optimize order routing
measure fill rate
commit execution branch
```

Merge after validation.

---

# 10. System Model: Producer / Consumer

Research layer:

Producer

Execution layer:

Consumer

Research produces alpha.

Execution converts alpha into realized profit.

---

# 11. Goal

Build a deterministic, scalable, and production-grade passive trading system where:

- alpha generation is independent
- execution is optimized independently
- system evolves safely without regressions

This architecture enables institutional-grade reliability and scalability.

---

END OF DOCUMENT

