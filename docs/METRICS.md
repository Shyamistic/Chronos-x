# Metrics Philosophy

ChronosX is intentionally conservative about what it measures and displays. The system prioritizes **truthful, realized metrics** over visually impressive but misleading dashboards. [web:18][web:20]

---

## Reported Metrics

ChronosX reports:

- **Realized PnL only**  
  - PnL for **closed** trades with actual fills and costs  
  - No mark‑to‑market extrapolations

- **Closed trades only**  
  - Open positions are tracked for risk and exposure  
  - They do not contribute to PnL until exits are executed

- **Execution metrics**  
  - Slippage versus reference price  
  - Latency between governance approval and exchange fill  
  - Fill ratios and partial fills [web:9]

This guarantees that what the system reports can be independently verified from execution records.

---

## What ChronosX Intentionally Avoids

ChronosX does **not** calculate or display:

- Unrealized PnL  
- Hypothetical “would‑have‑been” performance  
- Curve‑fit backtest charts

Unrealized PnL:

- Encourages over‑optimization  
- Hides exit, liquidity, and execution risk  
- Misleads judges and investors about real capital risk [web:12][web:18]

By design, ALPHA mode will show **zero realized PnL** until exit logic is enabled and validated.

---

## Post‑ALPHA Metrics

Once exits and full trade life‑cycles are enabled, ChronosX will extend metrics to include:

- Full trade‑level PnL attribution  
  - Strategy alpha versus execution cost (slippage, latency, fees) [web:9][web:20]  
- Risk metrics  
  - Drawdowns, exposure by asset/strategy, stress scenarios  
- Governance and compliance metrics  
  - Rule hit rates, blocked orders, and override frequencies [web:11]

These additions will maintain the same principle: only report what can be backed by concrete fills and deterministic rules.